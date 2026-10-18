from typing import Any, TypeVar, List

T = TypeVar("T")


# 1. 使用 Any 的函数
def get_first_any(items: List[Any]) -> Any:
    return items[0]


# 2. 使用泛型的函数
def get_first_generic(items: List[T]) -> T:
    return items[0]


if __name__ == "__main__":
    names = ["alice", "bob"]

    # 场景 A：使用 Any
    res_any = get_first_any(names)
    # res_any.
    # 【实验】在这里输入 res_any.
    # 现象：IDE 彻底瞎了，不会提示 upper()、lower() 等字符串方法！

    # 场景 B：使用泛型
    res_gen = get_first_generic(names)
    # 【实验】在这里输入 res_gen.
    # 现象：IDE 精准提示所有字符串自带的方法！

# Any 放弃了类型检查，让 Python 变成了无拘无束的动态语言（容易埋雷）。
# 泛型 既保留了编写代码的灵活性，又保住了静态类型的智能提示和安全检查。