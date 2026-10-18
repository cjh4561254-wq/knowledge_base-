from fastapi import FastAPI

app = FastAPI()


# ✅ 推荐：即使现在不需要 await，以后可能需要
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # 现在很简单
    return {"user_id": user_id}

    # 以后如果要查数据库，直接加 await 就行
    # user = await db.query(user_id)
    # return user
