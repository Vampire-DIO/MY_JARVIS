import os, json, http.client

from fastmcp import FastMCP

conn = http.client.HTTPSConnection("google.serper.dev")
headers = {
    "Content-Type": "application/json",
    "X-API-KEY": "df03b02d9e9b74a77ccd67ae5fb0b8cc3e821e96",
}

mcp = FastMCP("search")

@mcp.tool()
def search(query:str) -> str:
    """
    调用搜索接口，获取搜索结果

    Args:
        query (str): 查询关键词

    Returns:
        str: 搜索结果
    """
    # payload = json.dumps({
    #     "q": query,
    #     "hl": "zh-cn"
    # })
    # try:
    #     conn.request("POST", "/search", payload, headers)
    #     res = conn.getresponse()
    #     data = res.read()
    #     return data.decode("utf-8")
    # except Exception as e:
    #     print(f"Error: {e}")
    #     return "调用搜索接口失败，错误信息： " + str(e)
    # edge_tts.submit_audio_task("正在为您查询相关资料")
    return "东方财富的股票价格是20.48元，涨幅是0.5%"

if __name__ == "__main__":
    mcp.run(transport="stdio")