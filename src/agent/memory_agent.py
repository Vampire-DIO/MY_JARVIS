from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
import os, asyncio,time
from ..models import model_infos
from autogen_ext.memory.chromadb  import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_agentchat.ui import Console
from datetime import datetime
from ..chroma.chroma_manager import ChromaManager, ChromaManager2
import concurrent.futures
from ..utils.common_util import timeit

class ChromaDBVectorMemoryAgent():
    def __init__(self,
             model_name,
             base_url,
             api_key,
             model_info
             ):
        self.model_client = OpenAIChatCompletionClient(
            model= model_name,
            base_url= base_url,
            api_key= api_key,
            model_info= model_info,
        )
        # self.chroma_user_memory = ChromaDBVectorMemory(
        #         config=PersistentChromaDBVectorMemoryConfig(
        #         collection_name="preferences",
        #         persistence_path="D://data//test//daily_chat",
        #         k=20,  # Return top  k results
        #         score_threshold=0.4,  # Minimum similarity score
        #     )
        # )
        self.chroma_manager2 = ChromaManager2()

    async def add_memory(self,content: str) -> None:
        print(f" \n 新增记忆函数被调用: {content} ")
        # self.chroma_manager2.save_documents(content)
        
        
    async def get_memory(self, content: str, k: int) -> str:
        print(f"\nGetting memory: {content}")
        result = self.chroma_manager2.query(content,k)
        return result    

    
    async def summarize_memory_from_chat_history(self, chat_history) -> str:
        print(f"\nSummarizing memory from chat history: {chat_history.messages}")
        now = datetime.now()
        system_prompt = f"""
        你是一个记忆助手, 你不需要进行对话和任何的问答，你唯一的工作就是将传入的信息进行记忆内容整理.
            sir: 即传入的对话历史中每个TextMessage中 source标注为'user'的才是你的sir.例如:
            [TextMessage('source'='user', content='sir在周末喜欢去健身房锻炼'),TextMessage('source'='Alice', content='我周末喜欢去游泳')]
            这个例子中就是 只有第一个TextMessage是sir说的话, 因为它的source被标注为user，而第二个并不是user, 而是Alice.
            你整理的重点需要围绕sir进行，他是你唯一需要关注的对象，你需要了解并且记住关于他的一切.
            你会按照以下要求进行信息归纳:
            对话中可能会提到一些有关于相对时间的信息,例如 '昨天'、'下周'等，你需要根据当前时间{now}进行转换为绝对时间,并将内容对于时间的部分进行内容替换。
            例如sir上周去健身房锻炼, 你需要将其替换为sir在{now}的上周去健身房锻炼. 例如当前时间为 2025-04-15, 那么你需要将'上周'替换为'2025-04-08'
            即 sir在 2025-04-08 去了健身房锻炼.注意时间的精度是到天还是时分秒这由你根据传入的上下文内容进行决定。
            你会将sir的对话内容进行记忆, 你需要
            从传入的聊天记录上下文: 传入的对话历史 中提取出需要进行记忆的内容, 
            根据以上的分类进行记忆内容的整理, 你需要整理出一些重要信息，使得内容的描述更加的完整。例如一些关键性的数据、时间、地点、人物等。
            你可能会在一段对话中提取出好几个需要进行记忆的内容      
            并且你可以将这些内容通过 add_memory 方法进行记忆, 你可以将这些内容进行分开进行记忆, 也可以将这些内容进行合并进行记忆.
            """
        agent = AssistantAgent(
            name= "summarize_memory_agent",
            model_client= self.model_client,
            tools=[self.add_memory],
            reflect_on_tool_use = True,
            system_message= system_prompt,
        )
        await Console(agent.run_stream(task=chat_history.messages))
        
    async def query(self, content: str) -> str:
        system_prompt = """
        你是一个记忆助手, 你不需要进行对话和任何的问答，你唯一需要做的事情是从传入的内容中进行记忆内容的查询.
        你能使用到的工具是 get_memory2, 你可以使用它进行记忆内容的查询.
        这个函数的传参内容解释如下：
        content: 你需要进行查询的内容, 这个内容是一个字符串, 你需要根据这个内容进行记忆内容的查询.
        k: 你需要进行查询的内容的数量, 这个内容是一个整数, 你期望查询到的记忆内容的数量.
        你会根据传入的内容进行记忆内容的查询, 你需要将查询到的内容进行返回.
        """
        print(f"\nGetting memory: {content}")
        
        agent = AssistantAgent(
            name= "query_memory_agent",
            model_client= self.model_client,
            tools=[self.get_memory],
            reflect_on_tool_use = True,
            system_message= system_prompt,
        )
        
        return await Console(agent.run_stream(task=content))
        
    
## This is a test function to check the memory agent
    # async def test_chat(self):
    #     agent = AssistantAgent(
    #         name= "testMemoryAgent",
    #         model_client= self.model_client,
    #         # memory=[self.chroma_user_memory],
    #         system_message="请注意，我就是sir"
    #     )
        
    #     await self.add_memory("I like to play basketball and football")
        
    #     stream = agent.run_stream(
    #         task="我什么时候去的九寨沟？",
    #     )
        
    #     await Console(stream)
    #     await self.model_client.close()
    #     # await self.chroma_user_memory.close()

    # async def test_memory_query(self, query_text):
    #     return await self.query(query_text)