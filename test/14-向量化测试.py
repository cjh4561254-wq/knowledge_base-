from utils.embedding_utils import get_bge_m3_ef

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
model = get_bge_m3_ef()
embeddings = model.encode_documents(docs)# 向量化

#稀疏向量的数据获取
sparse = []
# embeddings["sparse"] 拿到的 是压缩稀疏矩阵
sparse_obj = embeddings["sparse"]# 拿到稀疏向量   CSR格式

# for循环 csr 格式 还原成  key：vlaue  还原成  稀疏矩阵
for i in range(len(docs)):
    sparse_data = sparse_obj.data[
        sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]
    ].tolist()

    sparse_indices = sparse_obj.indices[
        sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]
    ].tolist()

    sparse_dict = {
        k:v for k, v in zip(sparse_indices, sparse_data)
    }

    sparse.append(sparse_dict)

print(sparse) # 会得到键值对   标识:权重
#milvus数据库存储稀疏向量的格式要求
# {
#     key: "列"，
#     value: "值"
# }