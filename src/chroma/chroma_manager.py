import chromadb,os
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import hashlib
from langchain_community.embeddings import ZhipuAIEmbeddings
from zhipuai import ZhipuAI

zhipu_api_key = os.environ.get("ZHIPU_API_KEY")
class ChromaManager:
    def __init__(self, db_path="D://data//test//daily_chat"):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                allow_reset=True,
            )
        )
        self.embeddings = ZhipuAIEmbeddings(
            api_key=zhipu_api_key,
            model="embedding-3",
        )
        
        self.collection = self.client.get_or_create_collection(name="test2")

    def generate_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def add_documents(self, documents: list, metadatas: list = None):
        """
        插入文档并自动生成嵌入向量
        :param ids: 唯一标识列表
        :param documents: 文本内容列表
        :param metadatas: 元数据字典列表（可选）
        """
        ids = [self.generate_id(doc) for doc in documents]
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas if metadatas else [{}]*len(ids)
        )
        
    def query(self, query_text: str, n_results: int = 5, where: dict = None):
        """
        语义相似度搜索
        :param query: 查询文本
        :param n_results: 返回结果数
        :param where: 元数据过滤条件（网页5）
        :return: 结构化搜索结果
        """
        print(f"chroma_manager query 方法调用，查询文本：{query_text} ，返回结果数：{n_results}，过滤条件：{where}")
        
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        
    def update_document(self, doc_id: str, new_doc: str = None, new_metadata: dict = None):
        """
        更新文档内容或元数据（网页2）
        :param doc_id: 要更新的文档ID
        :param new_doc: 新文本内容（可选）
        :param new_metadata: 新元数据（可选）
        """
        update_data = {"ids": [doc_id]}
        if new_doc: update_data["documents"] = [new_doc]
        if new_metadata: update_data["metadatas"] = [new_metadata]
        
        self.collection.update(**update_data)
        

# if __name__ == "__main__":
#     # 初始化管理器
#     mgr = ChromaManager()
    
#     # 插入示例数据
#     # mgr.add_documents(
#     #     documents=["人工智能发展现状", "量子计算最新突破"],
#     #     metadatas=[
#     #         {"category": "AI", "date": "2025-04-01"},
#     #         {"category": "Quantum", "date": "2025-04-10"}
#     #     ]
#     # )
    
#     # 执行语义搜索
#     results = mgr.query("泸沽湖",1)
    
#     print(f"搜索结果：{results}")
    
    
import chromadb
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from ..utils.common_util import timeit

zhipu_api_key = os.environ.get("ZHIPU_API_KEY")


class ChromaManager2:
    
    def __init__(self, db_path="D://data//test2//daily_chat"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="test2")
        self.embeddings = ZhipuAIEmbeddings(
            api_key=zhipu_api_key,
            model="embedding-3",
        )
        self.persist_path = db_path
        self.vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )
    
    @timeit
    def save_documents(self, texts):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        spliced_docs = text_splitter.split_text(texts)
        vector = Chroma.from_texts(
            texts=spliced_docs,
            embedding= self.embeddings,
            persist_directory= self.persist_path
        )
        vector.persist()
    
    @timeit
    def query(self, query_text: str, n_results: int = 5):
        docs = self.vector_store.similarity_search(query_text, k=n_results)
        print(docs)
        page_contents = []
        for doc in docs:
            page_contents.append(doc.page_content)    
        return page_contents
    
    
if __name__ == "__main__":

    memory = ChromaManager2()
    
    print(memory.query("老板什么时候去的九寨沟？",1))
