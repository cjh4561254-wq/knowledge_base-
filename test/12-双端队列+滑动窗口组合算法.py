from collections import deque
import time


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests  # 窗口内最大允许次数（3次）
        self.window_seconds = window_seconds  # 滑动窗口的总时长（10秒）
        self.request_times = deque()  # 双端队列：专门存成功通过的时间戳

    def access_gate(self, user_name: str):
        current_time = time.time()
        print(f"\n--- 游客【{user_name}】在第 {current_time:.2f} 秒尝试刷闸机 ---")

        # 核心步骤 1：清理队头过期的元素
        # 如果当前时间 和 队头（最早的那个成功者）的时间差，已经大于等于 10 秒了，说明他已经不在当前的监控窗口内了，直接踢出队列
        while self.request_times and (current_time - self.request_times[0] >= self.window_seconds):
            expired_time = self.request_times.popleft()
            print(f"[清理] 检查到第 {expired_time:.2f} 秒通过的记录已过期，踢出窗口，释放1个名额")

        # 核心步骤 2：判断当前窗口内的总请求数是否达到了上限
        if len(self.request_times) >= 3:
            # 计算需要死等多久 = 10秒窗口总量 - 最早那个人已经耗掉的时间
            sleep_duration = self.window_seconds - (current_time - self.request_times[0])# 还要等多久
            if sleep_duration > 0:
                print(f"[限速拦截] 10秒内已经进去了 {len(self.request_times)} 个人！达到上限！")
                print(f"[强制等待] 闸机锁死！游客【{user_name}】必须原地等待 {sleep_duration:.2f} 秒...")

                # 阻塞死等
                time.sleep(sleep_duration)

                # 等完回来后，时间已经变了，需要更新当前时间并再次清理可能过期的数据
                current_time = time.time()
                while self.request_times and (current_time - self.request_times[0] >= self.window_seconds):
                    self.request_times.popleft()

        # 核心步骤 3：成功放行，记录当前时间戳到队尾
        self.request_times.append(current_time)
        print(f"✅ [放行成功] 游客【{user_name}】通过闸机！当前10秒窗口内共有 {len(self.request_times)} 人。")


# ==================== 测试运行 ====================
if __name__ == "__main__":
    # 初始化限速器：10秒内最多 3 次
    limiter = RateLimiter(max_requests=3, window_seconds=10)

    # 模拟疯狂涌入的游客
    tourists = ["张三", "李四", "王五", "赵六", "钱七"]#。。。。。

    for idx, name in enumerate(tourists):
        # # 前 4 个人几乎同时涌入（每隔 0.5 秒来一个）
        # if idx < 4:
        #     time.sleep(0.5)
        # else:
        #     # 最后一个钱七隔 2 秒再来
        #     time.sleep(2)
        time.sleep(0.5)

        limiter.access_gate(name)