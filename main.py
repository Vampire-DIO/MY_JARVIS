# from src.agent.memory_agent import ChromaDBVectorMemoryAgent
from src.agent.search_agent import SearchAgent
import asyncio,os,concurrent.futures
from src.models import model_infos
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat, Swarm
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from src.tts import edge_tts

import datetime
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from src.vtt.sence_voice import parse_audio, record_audio
from src.vtt import sence_voice as sv
import threading,time
from src.common import base_constants
api_key = os.environ.get("DPSK_API_KEY")


# memory_agent = ChromaDBVectorMemoryAgent(
#      model_name= "deepseek-chat",
#     base_url= "https://api.deepseek.com",
#     api_key= api_key,
#     model_info= model_infos.deepseek_model_info,
# )

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


model_client = OpenAIChatCompletionClient(
            model= "deepseek-chat",
            base_url= "https://api.deepseek.com",
            api_key= api_key,
            model_info= model_infos.deepseek_model_info,
)
async def tts(text:str):
    file_name = await edge_tts.tts(text)
    edge_tts.play_audio(file_name)
    
planner = AssistantAgent(
    name= "planner",
    model_client= model_client,
    system_message= """
    你是一个规划助手，你的工作是根据传入的内容进行规划，你会将规划的内容进行返回.
    当你认为已经从其他工具助手那里获取到了足够的信息时, 你结果进行口播内容总结, 口播内容指的是总结的信息是需要生成语音进行播放的, 
    因此一些转义符、其他文本格式的特殊符号（例如markdown格式中标题的#号）就不需要输出了. 保证输出的内容是可以直接进行播放的内容.
    你将会使用 tts 方法进行语音播放, 该方法的参数是你需要进行播放的内容.
    使用 'TERMINATE' 来结束任务. """,
    handoffs=["search"],
    tools=[tts],
    reflect_on_tool_use= True
)

async def init_mcp_tools():
    fetch_mcp_server = StdioServerParams(command="uv", args=["--directory","D://project//python//jarvis//src//utils","run","search.py"])
    mcp_tool = await mcp_server_tools(fetch_mcp_server)
    return mcp_tool

async def search_agent_create():
    search = await SearchAgent.create(model_client=model_client)
    return search.agent

async def create_mini_agent():
    search_tools = await init_mcp_tools()
    
    mini_agent = AssistantAgent(
        name= "mini_agent",
        model_client= model_client,
        system_message= """
        你是一个小助手，你总是会将用户的提问进行拆分成多个小问题,
        然后将这些小问题进行逐一解决回答，你可以使用工具帮你获取到一些信息帮助你更好的解决问题
        你能使用到的工具有:
        1. search: 搜索助手, 你可以使用该助手进行搜索, 该助手的参数是你需要进行搜索的内容.
        当你觉得已经得到最终答案后，你可以将答案进行总结, 然后使用 tts 方法进行语音播放, 该方法的参数是你需要进行播放的内容.
        使用 'TERMINATE' 来结束任务.
        请注意, 你的回答必须简洁明了, 你不能输出一些无用的内容, 你只需要输出你认为最重要的内容.
        你将会使用 tts 方法进行语音播放, 该方法的参数是你需要进行播放的内容.
        因此一些转义符、其他文本格式的特殊符号（例如markdown格式中标题的#号）就不需要输出了. 保证输出的内容是可以直接进行播放的内容.
        """,
        tools = search_tools,
        reflect_on_tool_use= True,
    )
    
    return mini_agent

async def main():
    search_agent = await search_agent_create()
    text_termination = TextMentionTermination("TERMINATE")
    termination = text_termination
    team = Swarm(
        participants=[planner, search_agent],
        termination_condition=termination,
    )
    begin =  datetime.datetime.now()
    
    task = "上周东方财富的股票行情如何？"
    await Console(team.run_stream(task=task))
    await model_client.close()
    print("任务执行耗时:", (datetime.datetime.now() - begin).total_seconds())
    
async def test_tts():
    text = "上周东方财富的股票行情如何？"
    file_name = await edge_tts.tts(text)
    # edge_tts.play_audio(file_name)
    
async def test_mini_agent(task):
    begin = datetime.datetime.now()
    mini_agent = await create_mini_agent()
    
    res = await mini_agent.on_messages(messages=[TextMessage(content=task, source="user")], cancellation_token=CancellationToken())
    
    print(f"\n res:  {res.chat_message.content} \n")
    await tts(res.chat_message.content)
    await model_client.close()
    print("任务执行耗时:", (datetime.datetime.now() - begin).total_seconds())
    
async def test_speak():
    mini_agent = await create_mini_agent()
    while(1):
        record_audio()
        task = parse_audio()
        res = await mini_agent.on_messages(messages=[TextMessage(content=task, source="user")], cancellation_token=CancellationToken())
        print(f"\n res:  {res.chat_message.content} \n")
        await tts(res.chat_message.content)
        
if __name__ == "__main__":
    # asyncio.run(main())
    # asyncio.run(main())
    # asyncio.run(test_speak())
    # while True:
    #     task = input("请输入任务:")
    #     if task == "exit":
    #         break

    #     asyncio.run(test_mini_agent(task))
    audio_thread = threading.Thread(target=sv.audio_recorder)
    audio_thread.start()
    while base_constants.IS_RUNNING:
        time.sleep(2)
        
    print("退出中...")
        

