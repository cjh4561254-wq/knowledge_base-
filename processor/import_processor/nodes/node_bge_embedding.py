# processor/import_processor/nodes/node_bge_embedding.py
"""
向量化节点：把切片（chunks）转成 BGE-M3 的 dense + sparse 双向量。

设计要点（为什么这样写）：
1. 惰性加载：本节点只在模块顶层 import generate_embeddings（纯函数引用），
   模型权重由 utils.embedding_utils 内部的首次调用惰性单例加载。
   这样导入整条 LangGraph 时不会在 import 期吃掉 ~2GB 内存 —— 只有真正跑到本节点才加载。
2. 批次大小走配置：batch_size = get_config().embedding_batch_size（当前 8），
   不再硬编码为 3。理由：CPU 版 torch 下 3 条/批会让几千条切片的向量化慢到不可用，
   而 8 是 embedding_utils 与本项目实测在内存安全与吞吐之间的折中；
   把批大小收敛到唯一配置源，主控改一次 .env/config 就能全局调优，避免各处写死漂移。
3. 严格对齐校验：embedding_utils 契约保证 dense/sparse 与输入 texts 一一对应，
   一旦出现长度不一致，立即抛 EmbeddingError 终止 —— 绝不静默错位。
   错位的向量会把 A 切片的语义绑到 B 切片上，污染整个知识库且极难排查。
4. 字段命名与下游一致：写入 chunk["dense_vector"] / chunk["sparse_vector"]，
   与 node_import_milvus.py 读取的键完全一致（该节点校验这两个字段存在并据此建集合 schema）。
"""

import json
import logging
import time
from typing import Dict, List

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.config_base import get_config
from processor.import_processor.exceptions import EmbeddingError, StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name: str = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：BGE-M3文本向量化处理
        流程总览：
            1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
            2. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
            3. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用

        必要参数：chunks
        更新参数：chunks字段新增dense_vector/sparse_vector

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：输入数据校验
        chunks = self._step_1_validate_input(state)

        # 本节点是整条导入链路的耗时大户（CPU 上几千条切片可能要几十分钟），
        # 处理前打点：让前端进度条 / 日志能看出卡在向量化这一步
        self.logger.info("向量化开始：待处理切片 %d 条", len(chunks))
        start_ts = time.perf_counter()

        # 步骤2：批量生成双向量，为切片绑定向量字段
        output_data = self._step_2_generate_embeddings(chunks)

        elapsed = time.perf_counter() - start_ts
        # 只记录条数/维度/耗时，绝不打印向量本身（1024 维 float 打进日志会刷屏且泄露无意义）
        self.logger.info(
            "向量化完成：切片 %d 条，耗时 %.2fs（平均 %.3fs/条）",
            len(output_data), elapsed, elapsed / max(1, len(output_data)),
        )

        # 步骤3：更新全局状态，将带向量的chunks回传下游
        state['chunks'] = output_data
        return state

    def _step_1_validate_input(self, state: ImportGraphState) -> List[Dict]:
        """
        步骤 1：输入数据有效性校验
        核心作用：
            1. 从全局状态提取待向量化的chunks切片列表
            2. 严格校验chunks类型和非空性，无有效数据则终止向量化
        参数：
            state: ImportGraphState - 流程全局状态对象
        返回：
            List[Dict] - 校验通过的文本切片列表
        异常：
            若chunks非列表/为空，抛出StateFieldError，终止当前向量化流程
        """

        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        # 逐条保证是 dict：非 dict 元素（如 None/字符串）后续 ['content'] 取值会抛裸 TypeError，
        # 这里提前用项目异常拦截，错误信息才能定位到具体下标
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise StateFieldError(
                    field_name="chunks",
                    message=f"chunks[{idx}] 不是字典，无法向量化",
                    expected_type=dict,
                )

        # item_name 容错：该字段由上游主体识别节点写入，硬下标会强依赖「主体识别先跑过」，
        # 单节点调试或图被改动跳过该节点时直接 KeyError。这里统一统计缺失条数并只打一条
        # 汇总 WARNING（逐条告警会刷屏），后续 _build_text 按契约用空串兜底，
        # 拼接形态保持「无商品名前缀 = 纯 content」，不改检索语义。
        missing_item_name = sum(1 for chunk in chunks if not (chunk.get("item_name") or "").strip())
        if missing_item_name:
            self.logger.warning(
                "%d/%d 个切片缺少 item_name 字段（上游主体识别节点可能未执行），"
                "按空串兜底向量化（无前缀拼接形态）。本警告整体只出现这一次；"
                "若非有意跳过主体识别，请检查图连线与上游节点产出。",
                missing_item_name, len(chunks),
            )

        return chunks

    def _build_text(self, chunk: Dict) -> str:
        """
        构造用于向量化的文本：商品名前置 + 换行 + 切片正文。

        为什么：item_name（商品名）是核心检索特征，拼在文本最前面能增强语义、
        让同商品切片在向量空间更聚集；同时下游按 content 检索正文语义。

        边界处理（空切片 / 缺字段 / 超长文本）：
          - item_name 或 content 为 None/缺失 → 视作空串，绝不抛 KeyError；
          - 两者都为空 → 仍返回空串交给模型（契约保证输出长度对齐，向量为空向量），
            这样空切片也有自己的向量槽位，不会因为被跳过而破坏 chunks 与向量的下标对应；
          - 超长文本 → 由 embedding_utils 内部按 BGE_MAX_LENGTH 截断，本节点不重复处理。
        """
        item_name = (chunk.get("item_name") or "").strip()
        content = chunk.get("content") or ""
        return f"{item_name}\n{content}" if item_name else content

    def _step_2_generate_embeddings(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        步骤 2: 批量生成向量（核心业务逻辑）
        核心逻辑：
            1. 分批处理：CPU 上一次性喂几千条会 OOM/极慢，按配置批大小切片，
               每批单独调用一次 generate_embeddings，批与批之间释放中间张量。
            2. 文本构造：将 item_name 和 content 拼接，增强语义（商品名作为核心特征前置）。
            3. 向量生成：调用模型批量生成 Dense（稠密）和 Sparse（稀疏）向量。
            4. 对齐校验：任何一批返回条数与输入条数不符立即抛 EmbeddingError，
               防止向量错位污染知识库。
        参数：
            chunks: List[Dict] 待向量化的文本切片列表
        返回：
            List[Dict]: 包含向量字段（dense_vector/sparse_vector）的文本切片列表
        """

        # 批大小走统一配置（当前=8）；兜底至少为 1，避免误配 0/负数导致 range 死循环
        batch_size = max(1, int(get_config().embedding_batch_size))
        total = len(chunks)
        self.logger.info("批处理向量化：batch_size=%d，总批次=%d", batch_size, (total + batch_size - 1) // batch_size)

        # 初始化一个空列表
        output_data: List[Dict] = []
        # 记录 dense 维度，用于完成后打点（只记一次，不打印向量值）
        dense_dim = None

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            # 批量构造文本：texts ---> [item_name\ncontent, ...]（与 batch_chunks 下标一一对应）
            texts = [self._build_text(chunk) for chunk in batch_chunks]

            # 批量生成向量（首次调用会在此处触发模型惰性加载，属预期行为）
            vectors = generate_embeddings(texts)

            if not isinstance(vectors, dict):
                raise EmbeddingError(
                    message=f"generate_embeddings 返回类型异常，期望 dict 实得 {type(vectors).__name__}",
                    node_name=self.name,
                )

            dense_vectors = vectors.get("dense")   # 稠密向量 list[list[float]]
            sparse_vectors = vectors.get("sparse")  # 稀疏向量 list[dict[int, float]]

            # 严格对齐校验：返回条数必须与输入条数一致，否则静默错位会污染整个库
            if dense_vectors is None or sparse_vectors is None:
                raise EmbeddingError(
                    message="generate_embeddings 返回缺失 dense 或 sparse 字段",
                    node_name=self.name,
                )
            if len(dense_vectors) != len(texts) or len(sparse_vectors) != len(texts):
                raise EmbeddingError(
                    message=(
                        f"向量条数与输入不一致（输入 {len(texts)} 条，"
                        f"dense {len(dense_vectors)} 条，sparse {len(sparse_vectors)} 条），"
                        f"已终止以避免向量错位污染知识库"
                    ),
                    node_name=self.name,
                )

            # 逐条把双向量绑回对应切片（此处下标 j 已与 texts/batch_chunks 严格对齐）
            for j, chunk in enumerate(batch_chunks):
                # 不修改上游传入的 dict（共享引用），拷贝一份再写入向量字段
                item = dict(chunk)
                item["dense_vector"] = dense_vectors[j]
                item["sparse_vector"] = sparse_vectors[j]
                output_data.append(item)

            if dense_dim is None and dense_vectors and isinstance(dense_vectors[0], (list, tuple)):
                dense_dim = len(dense_vectors[0])

            self.logger.info(
                "已向量化第 %d-%d 条（共 %d 条）",
                i + 1, min(i + len(batch_chunks), total), total,
            )

        # 完成后记录维度（仅维度数字，不打印向量内容）
        if dense_dim is not None:
            self.logger.info("稠密向量维度：%d，稀疏向量类型：dict[int,float]", dense_dim)

        return output_data


if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\output\hak180产品安全手册\state.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state.get("chunks")
    }

    # 执行核心处理流程
    node_bge_embedding = NodeBGEEmbedding()
    result = node_bge_embedding(init_state)

    vector_json_path = r"D:\output\hak180产品安全手册\state_vector.json"

    with open(vector_json_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False 确保中文字符及特殊标点正常显示，indent=4 让高维向量的数字排版美观
        json.dump(result, f, ensure_ascii=False, indent=4)

    logging.getLogger().info(f"💾 混合向量化成功！包含 dense/sparse 向量的数据已写入: {vector_json_path}")
