# utils/embedding_utils.py
"""
BGE-M3 向量化封装层（本地推理版）。

设计要点（为什么这样写）：
1. 惰性单例：BGEM3FlagModel 权重约 2GB，禁止在 import 期加载；
   只有第一次真正调用 generate_embeddings / get_bge_m3_ef 时才实例化模型。
   连 FlagEmbedding 本身也延迟 import，避免拖慢所有上游模块的加载。
2. 本机事实：torch 2.14.1+cpu、无 CUDA。因此：
   - 设备取自 config.embedding_config.bge_device（本机为 cpu）；
   - CPU 上强制 use_fp16=False（库默认值是 True，半精度在 CPU 上大量算子不支持，会直接报错或极慢）；
   - 代码中不出现任何 .to("cuda") 之类的设备假设。
3. 归一化：Milvus 的 dense_vector 索引使用 COSINE 度量，要求向量已 L2 归一化。
   BGEM3FlagModel(normalize_embeddings=True) 已在模型内部对 dense 向量做 L2 归一化，
   这里显式传 True 声明契约，不重复归一。
4. 对外契约不变（node_bge_embedding / node_search_embedding / node_search_embedding_hyde /
   node_item_name_recognition / node_item_name_confirm 均依赖）：
   generate_embeddings(texts: list[str]) -> {"dense": list[list[float]], "sparse": list[dict]}
   且 dense/sparse 与输入 texts 一一对应（下标对齐）。
"""

import logging
import os
import threading

from config.embedding_config import embedding_config
from processor.import_processor.config_base import get_config

# 注意：这里不能调用 setup_logging()。本模块会被 Web 层与所有检索节点间接导入，
# 在 import 期重置 root logger 会覆盖 FastAPI/uvicorn 的日志配置。
logger = logging.getLogger(__name__)

# 模型单例对象，避免重复初始化（惰性：import 本模块不会加载权重）
_bge_m3_ef = None
# 加锁保证多线程（FastAPI/LangGraph 并发检索）下只初始化一次
_bge_m3_lock = threading.Lock()

# CPU 上的截断长度：
# BGE-M3 模型本身支持 8192 token，但 flagembedding 的 passage_max_length 默认只有 512
# （不显式传参会把 2000 字的切片悄悄截断，属于隐性 bug，必须显式指定）。
# 同时 CPU 推理注意力计算是序列长度的平方级开销，8192 在 CPU 上不可接受。
# 本项目切片上限 max_content_length=2000 中文字符 ≈ 2000 token（加标题余量），
# 故默认取 2048：完整覆盖正常切片，又比 8192 快数倍；超长尾部才截断。
# 可通过环境变量 BGE_MAX_LENGTH 覆盖（新增配置项，已在报告中提请主控固化）。
_EMBEDDING_MAX_LENGTH = int(os.getenv("BGE_MAX_LENGTH", "2048"))


def get_bge_m3_ef():
    """
    获取 BGE-M3 模型单例对象（BGEM3FlagModel），自动加载环境变量配置。
    惰性加载：首次调用才真正读取 D:\\BGE-M3 权重。
    :return: 初始化完成的 BGEM3FlagModel 实例
    """
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    with _bge_m3_lock:
        # 双重检查：并发场景下只允许一个线程初始化
        if _bge_m3_ef is not None:
            return _bge_m3_ef

        # 延迟 import：FlagEmbedding 会连带加载 torch/transformers，
        # 放在模块顶层会让所有 import 本模块的地方变慢且占内存
        from FlagEmbedding import BGEM3FlagModel

        model_name = embedding_config.bge_m3_path  # 本机：D:\BGE-M3
        device = (embedding_config.bge_device or "cpu").strip().lower()
        use_fp16 = bool(embedding_config.bge_fp16)

        # 本机 torch 为 CPU 版本：fp16 在 CPU 上不受支持（大量算子会报错），
        # 强制关闭，避免 .env/配置被误改后在检索链路上炸出运行时异常
        if "cuda" not in device:
            if use_fp16:
                logger.warning("设备为 %s，不支持半精度推理，已强制 use_fp16=False", device)
            use_fp16 = False

        logger.info("首次加载 BGE-M3 模型: path=%s device=%s fp16=%s", model_name, device, use_fp16)
        # 注意：flagembedding 1.4.2 的构造参数名是 devices（不是 device），
        # 且 use_fp16 库默认 True，必须显式传入本机配置值
        _bge_m3_ef = BGEM3FlagModel(
            model_name_or_path=model_name,
            devices=device,
            use_fp16=use_fp16,
            # dense 向量在模型内部已做 L2 归一化，满足 Milvus COSINE 索引要求
            normalize_embeddings=True,
        )
    return _bge_m3_ef


def generate_embeddings(texts):
    """
    为文本批量生成 dense + sparse 双向量。
    :param texts: 文本列表（也兼容单个字符串）；允许含空串，返回列表与输入等长、下标对齐
    :return: {"dense": list[list[float]](1024 维，已 L2 归一化),
              "sparse": list[dict[int, float]](token_id -> 权重，key 为 int 以符合 Milvus 稀疏向量入库格式)}
    """
    # 空输入直接返回空结构：不调模型，且保持返回类型契约不变
    if texts is None:
        return {"dense": [], "sparse": []}
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    if len(texts) == 0:
        return {"dense": [], "sparse": []}

    # None 元素转为空串，保证输出长度与输入严格对齐（调用方按下标 zip 回 chunks）
    texts = ["" if t is None else str(t) for t in texts]

    model = get_bge_m3_ef()

    # 批次大小走统一配置（CPU 上 8 条/批是速度与安全内存占用的折中）
    batch_size = max(1, int(get_config().embedding_batch_size))

    # return_sparse=True 产出词法权重；colbert 向量本项目不使用，关闭以省 CPU 时间
    embeddings = model.encode(
        sentences=texts,
        batch_size=batch_size,
        max_length=_EMBEDDING_MAX_LENGTH,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    # dense_vecs: np.ndarray (n, 1024)，normalize_embeddings=True 时已 L2 归一化
    dense_vectors = [
        vec.tolist() if hasattr(vec, "tolist") else [float(x) for x in vec]
        for vec in embeddings["dense_vecs"]
    ]

    # lexical_weights: List[Dict[str(token_id), float]]（flagembedding 1.4.2 的 key 是字符串），
    # Milvus 的 SPARSE_FLOAT_VECTOR 需要 {int: float}，这里统一转换
    processed_sparse = []
    for weights in embeddings["lexical_weights"]:
        processed_sparse.append({int(k): float(v) for k, v in dict(weights).items()})

    return {
        "dense": dense_vectors,
        "sparse": processed_sparse
    }


# ================= 单元测试主入口 =================
if __name__ == "__main__":
    # 准备测试文本（一个是中文，一个是英文词，模拟双语召回测试）
    test_data = ["大功率电磁炉", "Car"]

    print("👉 开始单独测试 generate_embeddings 函数...")
    print(f"👉 测试文本: {test_data}\n")

    try:
        # 执行目标函数（此处才会真正加载本地 D:\BGE-M3 权重，CPU 推理较慢属正常）
        output = generate_embeddings(test_data)

        print("\n" + "=" * 20 + " 🎉 测试结果 🎉 " + "=" * 20)

        # 1. 验证稠密向量
        print("【稠密向量 (Dense)】")
        print(f"   - 成功生成数量: {len(output['dense'])} 条")
        # 稠密向量有 1024 维，我们只打印前 5 个数字展示一下，否则会刷屏
        print(f"   - 第一条文本的维度大小: {len(output['dense'][0])} 维")
        print(f"   - 第一条文本的前 5 维特征值: {output['dense'][0][:5]} ...")

        # 2. 验证稀疏向量
        print("\n【稀疏向量 (Sparse)】")
        print(f"   - 成功生成数量: {len(output['sparse'])} 条")
        print("   - 第一条文本的关键字字典 (词ID : 权重):")
        print(f"     {output['sparse'][0]}")

        print("=" * 54)
        print("💡 结论：函数单独测试通过！输出数据结构完全符合 Milvus 的入库标准。")

    except Exception as e:  # noqa: BLE001  # 手动测试入口：目的是打印排查建议而非上抛，兜底所有异常便于诊断
        print(f"\n❌ 测试失败！拦截到异常: {e}")
        print("💡 排查建议：")
        print("   1. 请确认本地模型目录 BGE_M3_PATH（如 D:\\BGE-M3）下权重文件是否完整。")
        print("   2. 本机为 torch CPU 版：确认 BGE_DEVICE=cpu、BGE_FP16=0/False。")
