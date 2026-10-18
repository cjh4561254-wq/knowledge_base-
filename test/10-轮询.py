import time

# 1. 一开始，饭还没熟（False）
rice_ready = False
count = 0  # 记录我看了几次

# 2. 只要饭没熟，就一直看（while 就是“只要...就...”）
while rice_ready == False:
    count = count + 1 # 看了一次
    print(f"第 {count} 次去看饭熟了没...")

    # 模拟：我们假设走到第 3 次的时候，饭终于熟了
    if count == 3:
        rice_ready = True  # 标记变成 True，代表熟了
        print("💡 呀，饭熟了！不看了！")
    else:
        print("❌ 还没熟，等 1 秒再去...")
        time.sleep(1)  # 关键：歇 1 秒，别把自己累死

# 3. 跳出循环后，证明饭已经熟了
print("🍚 盛饭，开吃！")