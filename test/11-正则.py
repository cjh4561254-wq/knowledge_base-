import re


def get_simple_context_with_re(text: str, keyword: str, context_len: int = 4):
    """
    使用 re 正则模块寻找关键词，并截取前后各 4 个字
    """
    # 1. re.compile: 将要查找的词编程正则模式对象
    # re.escape 是为了防止你的关键词里有 . 或 ? 这种会干扰正则的特殊符号
    pattern = re.compile(re.escape(keyword))

    # 2. pattern.search：让对象在文本里拉网式搜索
    # 它如果找到了，会返回一个“匹配结果对象（Match Object）”，我们叫它 match
    match = pattern.search(text)
    if not match:
        print("没找到关键词")
        return

    # 3. match.span()：它直接把起点和终点同时吐出来(也就是苹果的开始位置和结束位置)！ tart = ... 和 end = ...
    start, end = match.span()
    # print(start, end) 打印观察

    # 本质就是字符串切片
    pre_text = text[max(0, start - context_len): start]  # 我喜欢吃
    post_text = text[end: min(len(text), end + context_len)]

    print(f"【前文】: '{pre_text}'")
    print(f"【后文】: '{post_text}'")

    return pre_text, post_text


if __name__ == "__main__":
    # 测试同样的句子
    a = get_simple_context_with_re("派很好吃，我喜欢吃苹果，你呢？", keyword="苹果", context_len=4)
    print(a)