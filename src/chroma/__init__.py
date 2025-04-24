from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import os

class ChromaClient:
    def __init__(self, collection_name: str = "default"):
        self.embedding = OpenAIEmbeddings(
            model="text_embedding",
            openai_api_key="f9244a5791ed29b46e8aa24c39d7e97a.RWHfQK1MTttZgDND",
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/embeddings"
        )
        self.client = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到集合"""
        return self.client.add_documents(documents)

    def delete_documents(self, ids: List[str]) -> None:
        """从集合中删除文档"""
        self.client.delete(ids)

    def query(self, query: str, k: int = 5) -> List[Document]:
        """查询相似文档"""
        return self.client.similarity_search(query, k=k)

    def get_collection_stats(self) -> dict:
        """获取集合统计信息"""
        return self.client.get()

    def clear_collection(self) -> None:
        """清空整个集合"""
        self.client.delete_collection()
