# test/fastapi/backend_cors.py

from fastapi import FastAPI

import uvicorn
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# # ⚠️ 注意：这里先故意不配置 CORS！测试不跨域的效果
# # ✅ 配置 CORS（允许跨域）
app.add_middleware(
    CORSMiddleware,# 跨域中间件
    allow_origins=["*"],  # 允许所有来源（开发环境方便测试）
    allow_credentials=True, # 允许携带 Cookie
    allow_methods=["*"],  # 允许所有 HTTP 方法（GET, POST, PUT, DELETE 等）
    allow_headers=["*"],  # 允许所有请求头
)

@app.get("/api/data")
async def get_data():
    return {"message": "这是后端数据", "status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)

# uv run uvicorn test.import.fastapi.backend_cors:app --reload --port 8001