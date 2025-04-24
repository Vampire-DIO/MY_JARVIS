import os,threading
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from ..models import  model_infos
from ..tts import edge_tts
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from ..common import base_constants

api_key = os.environ.get("DPSK_API_KEY")
model_client = OpenAIChatCompletionClient(
            model= "deepseek-chat",
            base_url= "https://api.deepseek.com",
            api_key= api_key,
            model_info= model_infos.deepseek_model_info,
)



async def stop_audio_thread():
    print("正在关闭中...")
    base_constants.IS_RUNNING = False

mini_agent = AssistantAgent(
            name= "mini_agent",
            model_client= model_client,
            system_message= """
            你是一个超级管家叫做 Jarvis，你总是会称呼我为 sir
            请注意, 你的回答必须简洁明了, 你不能输出一些无用的内容, 你只需要输出你认为最重要的内容.
            你将会使用 tts 方法进行语音播放, 该方法的参数是你需要进行播放的内容.
            因此一些转义符、其他文本格式的特殊符号（例如markdown格式中标题的#号）就不需要输出了. 保证输出的内容是可以直接进行播放的内容.
            当你认为你的任务已经结束时, 你需要调用 stop_audio_thread 方法
            """,
            tools=[stop_audio_thread],
            reflect_on_tool_use= True,
        )

async def tts(text:str):
    print(f"tts输出接收到的文本内容: {text}")
    file_name = await edge_tts.tts(text)
    edge_tts.play_audio(file_name)

async def answer(task):
    print("正在思考问题中....")
    res = await mini_agent.on_messages(messages=[TextMessage(content=task, source="user")], cancellation_token=CancellationToken())
    print(f"\n res:  {res.chat_message.content} \n")
    await tts(res.chat_message.content)