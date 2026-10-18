# processor/import_processor/nodes/node_import_milvus.py
"""导入流水线终点节点：把带双向量的切片写入 Milvus 集合 kb_chunks。

设计要点（为什么这样写）：
1. 建表逻辑唯一化：Schema/索引只在 utils.milvus_utils.ensure_chunks_collection()
   里定义一份。节点内不再内联 create_schema —— 两处 schema 各自演化必然漂移，
   最终导致「检索端字段名/度量类型与写入端对不上」这类最难排查的 bug。
2. 字段契约（与 milvus_utils / node_document_split / node_item_name_recognition 对齐）：
   kb_chunks: chunk_id(自增主键) content title parent_title part file_title
              item_name sparse_vector dense_vector
   注意：全项目不存在 doc_id 字段，幂等删除依据是 file_title。
3. 插入前数据整形抽成纯函数 prepare_insert_rows()，可脱离 Milvus/模型单测：
   - dense 维度必须等于 EMBEDDING_DIM（本机 1024）；
   - sparse 必须是 {int: float} 且非空 —— 空文本会产生空 sparse，
     插入 SPARSE_FLOAT_VECTOR 字段服务端直接报错，必须在这里拦下；
   - content/向量缺失或为 None 的切片直接报错终止（一步一校验，不脏写）。
4. 分批插入：单批默认 50 条（可用环境变量 MILVUS_INSERT_BATCH_SIZE 覆盖），
   避免一次 gRPC 请求携带过多 1024 维向量导致消息体超限。
   任一批失败会明确报出「批次范围 + 原因」，绝不静默丢数据。
5. 幂等：同 file_title 的旧数据先按 in 过滤一次性删除再插入，重复导入结果一致。
6. 收尾：Milvus 2.5 插入后需 flush 落盘 + load_collection 保证立刻可检索。
"""
import json
import logging
import os
from typing import Any, Dict, List, Tuple

from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import MilvusError, StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.milvus_utils import build_in_filter, ensure_chunks_collection, get_milvus_client

# ==================== 常量（与 ensure_chunks_collection 的 schema 严格同步） ====================

# 单批插入条数：切片 ≈ 2000 字符正文 + 1024 维 dense + 变长 sparse，
# 50 条/批约几 MB，远低于 gRPC 默认消息上限，也足够摊薄 RPC 开销
_INSERT_BATCH_SIZE = max(1, int(os.getenv("MILVUS_INSERT_BATCH_SIZE", "50")))

# 各 VARCHAR 字段的字节上限（Milvus max_length 按 UTF-8 字节计，中文 1 字符占 3 字节）
_VARCHAR_LIMITS = {
    "title": 100,
    "parent_title": 100,
    "file_title": 100,
    "item_name": 100,
    "content": 65535,
}

# INT8 取值范围：part 超出会直接被服务端拒绝，兜底裁剪避免整批插入失败
_INT8_MIN, _INT8_MAX = -128, 127


def _clamp_utf8_bytes(value: str, max_bytes: int) -> str:
    """
    按 UTF-8 字节数截断字符串，且绝不截断半个多字节字符。
    为什么要截断而不是报错：标题类字段偶发超长是正常业务现象（长文件名/长标题），
    整条切片内容本身是有效的，因 1 个字节超限丢掉整个 chunk 得不偿失。
    """
    raw = value.encode("utf-8")
    if len(raw) <= max_bytes:
        return value
    cut = raw[:max_bytes]
    # 截断点可能落在多字节字符中间，逐字节回退到合法边界
    while cut:
        try:
            return cut.decode("utf-8")
        except UnicodeDecodeError:
            cut = cut[:-1]
    return ""


def prepare_insert_rows(chunks: List[Dict[str, Any]], expected_dim: int) -> List[Dict[str, Any]]:
    """
    插入前数据整形（纯函数，不触碰 Milvus，便于单测覆盖）。
    核心职责：
        1. 拦截脏数据：content/dense_vector/sparse_vector 缺失、None、空串、空 sparse
           —— 全部按切片下标报错上抛，禁止静默丢数据；
        2. 类型归一：sparse 的 key 强制转原生 int（JSON 反序列化后可能是 str/numpy 类型，
           gRPC 序列化 SPARSE_FLOAT_VECTOR 时只认 int key）；
        3. 字段补全与裁剪：parent_title/part 兜底、VARCHAR 按字节截断、
           移除可能残留的历史 chunk_id（自增主键冲突）；
        4. 只产出 schema 声明的 8 个字段，其余上游临时键不回写 Milvus。
    参数：
        chunks: 上游（切片→主体识别→向量化）产出的切片列表
        expected_dim: 稠密向量维度（应为 EMBEDDING_DIM，本机 1024）
    返回：
        List[Dict] - 与 chunks 一一对应（下标对齐，供主键回填）的待插入行
    异常：
        ValueError - 任一切片数据非法（错误信息中包含切片下标与原因）
    """
    rows: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        # ---------- 1. 正文校验：None/空串一律拦截 ----------
        content = chunk.get("content")
        if content is None or not str(content).strip():
            raise ValueError(f"切片[{idx}] content 为空，BGE-M3 空文本会产生空稀疏向量，无法写入 Milvus")
        content = str(content)

        # ---------- 2. 稠密向量校验：必须是等长的浮点列表 ----------
        dense = chunk.get("dense_vector")
        if not isinstance(dense, (list, tuple)) or len(dense) == 0:
            raise ValueError(f"切片[{idx}] dense_vector 缺失或为空")
        if len(dense) != expected_dim:
            raise ValueError(f"切片[{idx}] dense_vector 维度不符：实际 {len(dense)}，期望 {expected_dim}")
        try:
            dense = [float(x) for x in dense]
        except (TypeError, ValueError) as e:
            raise ValueError(f"切片[{idx}] dense_vector 含非数值元素：{e}") from e

        # ---------- 3. 稀疏向量校验：非空 dict，key 必须是 int ----------
        sparse = chunk.get("sparse_vector")
        if not isinstance(sparse, dict) or len(sparse) == 0:
            raise ValueError(f"切片[{idx}] sparse_vector 缺失/为空/类型错误（空稀疏向量会被 Milvus 拒绝）")
        try:
            sparse = {int(k): float(v) for k, v in sparse.items()}
        except (TypeError, ValueError) as e:
            raise ValueError(f"切片[{idx}] sparse_vector 键值类型非法（key 须可转 int）：{e}") from e

        # ---------- 4. 标量元数据补全 + 字节裁剪 ----------
        row: Dict[str, Any] = {"content": _clamp_utf8_bytes(content, _VARCHAR_LIMITS["content"])}
        for field_name in ("title", "parent_title", "file_title", "item_name"):
            value = chunk.get(field_name)
            # title/file_title/item_name 是必填 VARCHAR：缺失时补空串而不是报错，
            # 保证「向量化成功但元数据略缺」的切片仍能入库（parent_title 本就有默认值 ""）
            row[field_name] = _clamp_utf8_bytes(str(value) if value is not None else "", _VARCHAR_LIMITS[field_name])

        # part：切分节点对超长章节会给真实序号，其余兜底为 0；裁剪到 INT8 范围
        try:
            part = int(chunk.get("part", 0) or 0)
        except (TypeError, ValueError):
            part = 0
        row["part"] = max(_INT8_MIN, min(_INT8_MAX, part))

        row["dense_vector"] = dense
        row["sparse_vector"] = sparse
        # 注意：不携带 chunk_id —— 主键由 Milvus auto_id 生成后回填；
        # 历史 chunk_id（重复导入残留）若带上会与自增主键冲突
        rows.append(row)

    return rows


class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化（流水线终点）
    """

    name: str = "node_import_milvus"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：Milvus切片数据入库主流程
        执行流程（串行执行，一步一校验，保证数据一致性）：
            1. 输入校验：验证切片有效性、向量字段完整性，校验向量维度
            2. 环境准备：连接Milvus，幂等确保集合存在（统一走 ensure_chunks_collection）
            3. 幂等清理：按 file_title 删除旧数据，避免重复存储
            4. 数据整形+分批插入：纯函数整形后批量入库，回填Milvus自增chunk_id
            5. 收尾：flush 落盘 + load_collection，保证数据立刻可检索（Milvus 2.5 要求）

        异常处理：
            任一步骤失败抛出异常，终止节点执行，保证数据不脏写

        必要参数：chunks
        更新参数：chunks字段回填chunk_id

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：输入数据有效性校验
        chunks, vector_dimension = self._step_1_check_input(state)

        # 步骤2：Milvus客户端连接+集合准备（幂等建表，Schema 唯一定义在 milvus_utils）
        client = self._step_2_prepare_collection(vector_dimension)

        # 运行一次或者无数次结果都一样：按 file_title 删旧重写
        # 步骤3：幂等性处理 - 清理同 file_title 旧数据
        self._step_3_clean_old_data(client, chunks)

        # 步骤4：插入前整形 + 分批插入 + 主键chunk_id回填
        updated_chunks, inserted_count = self._step_4_insert_data(client, chunks, vector_dimension)

        # 步骤5：flush + load 收尾，保证写入数据立刻可被检索
        self._step_5_finalize(client)

        # 步骤6：更新全局状态，将回填后的切片回传下游
        state["chunks"] = updated_chunks

        collection_name = milvus_config.chunks_collection
        self.logger.info(
            f"Milvus 写入完成：集合[{collection_name}]，本次插入 {inserted_count} 条切片（来源 {len(chunks)} 个chunk）")

        return state

    def _step_1_check_input(self, state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
        """
        步骤1：输入数据有效性校验
        核心校验项：
            1. chunks非空且为列表类型
            2. 切片包含 dense_vector / sparse_vector 字段
            3. 向量维度必须等于配置 embedding_dim（BGE-M3 = 1024），
               维度不符说明上游模型/配置错位，直接终止，避免建出错误 dim 的集合
        参数：
            state: Dict[str, Any] - 流程状态对象，包含上游传入的chunks数据
        返回：
            Tuple[List[Dict], int] - (校验通过的切片列表, 稠密向量维度)
        异常：
            StateFieldError - 任一校验项不通过，终止入库，避免脏数据处理
        """

        # 校验1：chunks非空
        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        first_chunk = chunks[0]
        if not isinstance(first_chunk, dict):
            raise StateFieldError(field_name="chunks", message="chunks元素必须为dict", expected_type=dict)

        # 校验2：切片包含dense_vector字段
        if 'dense_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失dense_vector字段")

        # 校验3：切片包含 sparse_vector 字段
        if 'sparse_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失sparse_vector字段")

        # 校验4：向量维度与配置一致（全链路契约：embedding_utils 产出 1024 维）
        vector_dimension = len(first_chunk['dense_vector'])
        expected_dim = int(self.config.embedding_dim)
        if vector_dimension != expected_dim:
            raise StateFieldError(
                field_name="dense_vector",
                message=f"稠密向量维度不符：实际 {vector_dimension}，期望 {expected_dim}（检查 EMBEDDING_DIM 与向量化模型是否一致）",
            )

        return chunks, vector_dimension

    def _step_2_prepare_collection(self, vector_dimension: int):
        """
        步骤2：Milvus客户端连接+集合准备
        核心逻辑：
            1. 获取Milvus单例客户端
            2. 幂等确保集合存在：建表逻辑统一收敛到 utils.milvus_utils.ensure_chunks_collection，
               节点内不再内联 schema（历史上两处 schema 各自演化会漂移）
        参数：
            vector_dimension: int - 稠密向量维度（步骤1校验通过）
        返回：
            MilvusClient - 已连接、集合准备完成的客户端实例
        异常：
            MilvusError - 连接失败/建表失败
        """
        try:
            milvus_client = get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 连接失败: {e!s}")
            raise MilvusError(f"Milvus 连接失败: {e!s}") from e

        if not milvus_client:
            self.logger.error("Milvus 连接失败")
            raise MilvusError("Milvus 连接失败")

        try:
            # 集合存在则直接复用并 load；不存在则按统一 Schema 创建（幂等）
            return ensure_chunks_collection(client=milvus_client, vector_dimension=vector_dimension)
        except MilvusError:
            raise
        except Exception as e:
            self.logger.error(f"Milvus 集合准备失败[{milvus_config.chunks_collection}]: {e!s}", exc_info=True)
            raise MilvusError(f"Milvus 集合准备失败: {e!s}") from e

    def _step_3_clean_old_data(self, client, chunks: List[Dict[str, Any]]):
        """
        幂等清理
        基于本批切片涉及的所有 file_title 删除旧数据（一次性 in 过滤，多次导入结果一致）。
        为什么按全部去重 file_title 而不是只取第一个：state 理论上可能聚合多文件切片，
        只删第一个会留下其它文件的旧版本，产生脏重复。
        :param client: milvus客户端
        :param chunks: 本批待插入切片
        """
        file_titles = sorted({
            str(c.get("file_title", "")).strip()
            for c in chunks
            if str(c.get("file_title", "")).strip()
        })

        if not file_titles:
            self.logger.warning("本批切片均无 file_title，跳过幂等清理（可能出现重复数据）")
            return

        self._clear_chunks_by_file_titles(client, file_titles)

    def _clear_chunks_by_file_titles(self, client, file_titles: List[str]):
        """
        按 file_title 批量删除 kb_chunks 旧数据。
        过滤表达式统一用双引号字面量（build_in_filter 内部做转义），
        避免文件名含引号时表达式解析失败。
        """
        try:
            client.delete(
                collection_name=milvus_config.chunks_collection,
                filter=build_in_filter("file_title", file_titles),
            )
            self.logger.info(f"幂等清理：已删除集合[{milvus_config.chunks_collection}]中 {len(file_titles)} 个文件旧数据")
        except Exception as e:
            self.logger.error(f"Milvus 数据删除失败: {e!s}", exc_info=True)
            raise MilvusError(f"Milvus 数据删除失败: {e!s}") from e

    def _step_4_insert_data(self, client, chunks: List[Dict[str, Any]],
                            vector_dimension: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        步骤4：插入前整形 + 分批插入切片数据到Milvus + 主键回填
        核心逻辑：
            1. prepare_insert_rows 纯函数整形：拦截空文本/坏向量、sparse key 转 int、
               维度校验、VARCHAR 字节裁剪、剔除历史 chunk_id
            2. 分批插入：单批 _INSERT_BATCH_SIZE 条，控制单次 gRPC 消息体大小；
               失败时报出批次范围（第 m~n 条）与原因，终止流程，绝不静默丢数据
            3. 回填chunk_id：按全局下标对齐回填到原始切片，供下游/幂等重跑使用
        参数：
            client - MilvusClient实例
            chunks: List[Dict[str, Any]] - 待入库的切片列表
            vector_dimension: int - 稠密向量维度
        返回：
            Tuple[List[Dict], int] - (回填了chunk_id的切片列表, 实际插入条数)
        """
        rows = prepare_insert_rows(chunks, expected_dim=vector_dimension)

        collection_name = milvus_config.chunks_collection
        inserted_ids: List[Any] = []
        batch_size = _INSERT_BATCH_SIZE

        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            end = start + len(batch) - 1
            try:
                result = client.insert(collection_name=collection_name, data=batch)
            except Exception as e:
                # 明确报出批次范围与原因：哪一段没写进去一目了然，便于重跑定位
                self.logger.error(
                    f"Milvus 批量插入失败：集合[{collection_name}] 第 {start}~{end} 条切片，原因：{e!s}",
                    exc_info=True)
                raise MilvusError(
                    f"Milvus 批量插入失败（第 {start}~{end} 条，已成功 {len(inserted_ids)} 条）：{e!s}") from e
            inserted_ids.extend(result.get("ids", []))

        # 主键回填（行与切片下标严格一一对应）
        if len(inserted_ids) != len(chunks):
            raise MilvusError(
                f"Milvus 插入返回主键数量不符：插入确认 {len(inserted_ids)} 条，切片 {len(chunks)} 条")
        for idx, item in enumerate(chunks):
            item["chunk_id"] = inserted_ids[idx]

        return chunks, len(inserted_ids)

    def _step_5_finalize(self, client):
        """
        步骤5：写入收尾（Milvus 2.5 必需）
            flush：把 growing segment 中数据落盘，保证持久化；
            load_collection：保证新数据立刻可被检索（未 load 集合检索直接报错）。
        失败仅告警不抛穿：此时数据已插入成功（insert 有 ack），
        flush/load 偶发失败只影响可见性延迟，不值得让整条导入流水线重试重插。
        """
        collection_name = milvus_config.chunks_collection
        try:
            client.flush(collection_name)
        except Exception as e:
            self.logger.warning(f"Milvus flush 失败（数据已插入，可见性可能延迟）：{e!s}")
        try:
            client.load_collection(collection_name)
        except Exception as e:
            self.logger.warning(f"Milvus load_collection 失败（可能已加载）：{e!s}")


#测试
if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\output\hak180产品安全手册\state_vector.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "task_id": "local-test",
        "chunks": state.get("chunks")
    }

    # 执行核心处理流程
    node_import_milvus = NodeImportMilvus()
    result = node_import_milvus(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4, default=str))
