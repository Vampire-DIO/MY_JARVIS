from src.chroma import ChromaClient
from langchain_core.documents import Document


# 初始化客户端
client = ChromaClient("test_collection")

# 创建测试文档
documents = [
    Document(page_content="LangChain是一个强大的LLM应用开发框架", metadata={"source": "doc1"}),
    Document(page_content="DeepSeek提供了高质量的embedding模型", metadata={"source": "doc2"}),
    Document(page_content="Chroma是一个轻量级的向量数据库", metadata={"source": "doc3"})
]

# 添加文档
doc_ids = client.add_documents(documents)
print(f"添加文档成功，ID: {doc_ids}")

# 查询文档
query = "什么是LangChain?"
results = client.query(query)
print("\n查询结果:")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content} (来源: {doc.metadata['source']})")

# 获取集合统计
stats = client.get_collection_stats()
print(f"\n集合统计: {stats}")

# 清理测试集合
client.clear_collection()
print("\n已清理测试集合")
