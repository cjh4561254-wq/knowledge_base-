# \test\fastapi\bg_task.py
import asyncio

from fastapi import BackgroundTasks, FastAPI
import time

app = FastAPI()

# 定义一个模拟的耗时任务（方式1）
# ✅ 此处适合 CPU 密集型或阻塞操作
def write_log1(email: str, content: str):
    while True:
        print(f"异步任务正在执行...... 向 {email} 发邮件，内容是：{content}  {time.asctime()}")
        time.sleep(1)  # 模拟耗时操作
    #process_large_file()  # 同步的文件处理
    #run_cpu_heavy_task()  # CPU 计算


# 定义一个模拟的耗时任务（方式2）
# ✅ 此处适合 IO 密集型任务
async def write_log2(email: str, content: str):
    while True:
        print(f"异步任务正在执行...... 向 {email} 发邮件，内容是：{content}  {time.asctime()}")
        await asyncio.sleep(1)  # 模拟耗时操作
    # await send_email_async(email)  # 调用其他异步函数
    # await save_to_db_async(content)  # 异步数据库操作

@app.post("/send-task/{email}")
async def send_task(email: str, background_tasks: BackgroundTasks):
    # 1. 添加任务到后台队列
    background_tasks.add_task(write_log2, email, "你好")
    background_tasks.add_task(write_log1, email, "哈哈")
    # 2. 立即返回响应给用户，不需要等待 write_log 执行完毕
    return {"message": "异步任务已启动"}

#  background_tasks.add_task  添加后台任务

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uv run uvicorn test.import.fastapi.bg_task:app --reload