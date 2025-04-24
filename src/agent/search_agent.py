from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from ..models import model_infos
import os
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

class SearchAgent():
    
    async def init_mcp_tools(self):
        fetch_mcp_server = StdioServerParams(command="uv", args=["--directory","D://project//python//jarvis//src//utils","run","search.py"])
        mcp_tool = await mcp_server_tools(fetch_mcp_server)
        return mcp_tool
    
    @classmethod
    async def create(cls, model_client):
        """工厂方法模式创建SearchAgent实例"""
        instance = cls.__new__(cls)
        instance.name = "search"
        instance.model = model_client
        
        # 异步初始化MCP工具
        tools = await instance.init_mcp_tools()
        
        instance.agent = AssistantAgent(
            name="search",
            model_client=instance.model,
            system_message="""
            你是一个信息搜索助手，你的任务是根据用户的查询信息意图进行搜索，返回相关的搜索结果。你主要的能力是进行网络搜索.
            在你搜索得到结果后，你总是会将结果提交给planner
            
            """,
            tools=tools,
            reflect_on_tool_use=True,
            handoffs=["planner"]
        )
        return instance

    def __init__(self, *args, **kwargs):
        """禁止直接实例化，必须使用create工厂方法"""
        raise RuntimeError("请使用SearchAgent.create()方法创建实例")
