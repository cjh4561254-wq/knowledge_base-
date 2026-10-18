# test/fastapi/demo.py
import uvicorn
from fastapi import FastAPI
from tool.logger import logger

# 创建一个 FastAPI 应用实例
app = FastAPI()


# 摘要说明
@app.get("/", summary="第一个测试")
async def read_root():
    return {"Hello": "World"}


# 可以接受用户的参数
# 访问 http://127.0.0.1:8000/items/5?q=somequery
# item_id: 路径参数 (自动转为 int)
# q: 查询参数 (可选，默认 None)
@app.get("/items/{item_id}", summary="获取指定参数")
async def read_item(item_id: int, q: str | None = None):
    # 。。。 LLM
    # 查出来
    return {"item_id": item_id, "q": q}


# http://127.0.0.1:8000/items?skip=0&limit=10
# 接收? skip=? & limit = ?   page  size
@app.get("/items", summary="分页")
async def read_item(start: int = 0, size: int = 10):
    return {"start": start, "size": size}


from pydantic import BaseModel


# 定义数据模型
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None


# POST 请求接收 JSON 数据
@app.post("/items/", summary="类型检查")
async def create_item(item: Item):
    # item 已经是验证过的 Item 对象
    # 如果客户端传来的 price 是字符串 "abc"，FastAPI 会自动报错
    return {"name": item.name, "price": item.price, "is_offer": item.is_offer}


# 用户给你发请求  你给用户返回json

# 1、路由处理函数返回一个 Pydantic 模型实例，FastAPI 将自动将其转换为 JSON 格式，并作为响应发送给客户端：
@app.post("/items/return", summary="返回 Pydantic 模型实例")
async def create_item(item: Item):
    # LLM  结论
    return item


# 2、使用 HTTPException 抛出异常，返回自定义的状态码和详细信息。
# 以下实例在 item_id 为 42 会返回 404 状态码：
from fastapi import HTTPException


@app.delete("/items/{item_id}", summary="抛出异常")
async def read_item(item_id: int):
    if item_id == 42:# 只要你给我传个42  我主动跑信息
        raise HTTPException(status_code=404, detail="Item 找不到")
    return {"item_id": item_id}


from fastapi.responses import JSONResponse

@app.get("/api/user")
async def get_user():
    # 等价于直接 return {"name": "张三", "age": 20}（FastAPI 自动转 JSONResponse）
    return JSONResponse(
        content={"name": "张三", "age": 20},
        status_code=201,  # 可选，默认 200
        headers={"X-Custom-Header": "custom-value"}  # 可选，自定义响应头
    )






from fastapi.responses import FileResponse

@app.get("/download/excel")
async def download_excel():
    excel_path = r"D:\output\baby.xls"
    # 返回文件并指定下载文件名
    return FileResponse(
        path=excel_path,
        filename="baby.xls",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"# 我给你返回东西的类型
    )

from fastapi.responses import HTMLResponse

@app.get("/hello")
async def hello(name: str = "游客"):
    html_content = f"""
    <html>
        <body>
            <h1>你好，{name}！</h1>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)
from fastapi.responses import PlainTextResponse

"""
200
201
202
203
"""

@app.get("/text")
async def get_text():
    return PlainTextResponse(content="这是纯文本响应", status_code=200)


from fastapi.responses import RedirectResponse

@app.get("/old-path")
async def redirect_old_path():
    # 重定向到 /new-path，状态码 307 表示临时重定向
    return RedirectResponse(url="/new-path", status_code=307)

@app.get("/new-path")
async def new_path():
    return {"message": "这是新接口"}

from fastapi.responses import StreamingResponse
import asyncio

async def generate_stream():
    # 模拟流式输出（逐字返回）
    words = ["你", "好", "，", "这", "是", "流", "式", "响", "应"]
    for word in words:
        await asyncio.sleep(0.5)
        yield word.encode("utf-8")  # 流式输出需返回字节流
        # return 流式输出的一定不能用reutrn 全部内容结束 一次性输出
        # yield 拿到一个字 输出一个字


from fastapi.responses import Response

@app.get("/custom")
async def custom_response():
    # 返回二进制数据，指定自定义 MIME 类型
    return Response(
        content="<h1>纯文本</h1>",
        # media_type="text/text",
        media_type="text/html",
        status_code=200)


@app.get("/stream")
async def stream_response():
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    """服务启动入口：本地开发环境直接运行"""
    logger.info("File Import Service 服务启动中...")
    # 启动uvicorn服务，绑定本地IP和8000端口，关闭自动重载（生产环境建议用workers多进程）
    uvicorn.run(
        app=app,
        host="127.0.0.1",  # 仅本地访问，生产环境改为0.0.0.0（允许所有IP访问）
        port=8000  # 服务端口
    )
