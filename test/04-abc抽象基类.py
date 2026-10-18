from abc import ABC, abstractmethod


# 1. 抽象基类
class BaseWorker(ABC):
    def __call__(self, text: str):
        print("--- [日志] 开始干活啦 ---")  # 统一的公共逻辑

        result = self.do_work(text)  # 调用具体的工作（由子类决定）

        print("--- [日志] 干完收工啦 ---")  # 统一的公共逻辑
        return result
    # 一旦一个类里包含了 @abstractmethod，这个类就变成了“纯概念”（抽象类）。你不能直接创建它的对象，它只能作为骨架被别人继承。
    # 必须实现被 @abstractmethod 装饰的方法 （TypeError）
    @abstractmethod
    def do_work(self, text: str):
        """所有员工【必须】实现这个方法，怎么干活自己写"""
        pass


# 2. 员工 A：负责把字变大写（具体实现类）
class UpperWorker(BaseWorker):
    def do_work(self, text: str):
        return text.upper()


# 3. 员工 B：负责把字倒过来（具体实现类）
class ReverseWorker(BaseWorker):
    def do_work(self, text: str):
        return text[::-1]


if __name__ == "__main__":
    # 让大写员工干活
    worker_a = UpperWorker()
    print("结果:", worker_a("hello"))# 等价 worker_a.__call__("hello")

    print("\n" + "=" * 30 + "\n")

    # 让倒序员工干活
    worker_b = ReverseWorker()
    print("结果:", worker_b("hello"))
