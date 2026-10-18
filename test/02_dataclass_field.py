# from dataclasses import dataclass, field
#
#
# @dataclass
# class User:
#     # 1. 必需字段：没有默认值，创建实例时必须提供
#     username: str # ✅
#     # 2. 简单默认值：适用于不可变类型 (str, int, float等)
#     age: int = 18
#     # 3. 动态默认值：适用于可变类型 (list, dict等)，防止实例间共享状态
#     tags: list = field(default_factory=list)
#
# # u1 = User()
# # --- 开始测试 ---
#
# # print("--- 测试 1: 只提供必需字段 ---")
# # user1 = User("zhangsan")
# # print(f"user1: {user1}")
# # # 输出: user1: User(username='Alice', age=18, tags=[])
#
# # print("\n--- 测试 2: 提供所有字段 ---")
# # user2 = User("Bob", 25, ["developer", "pythonist"])
# # print(f"user2: {user2}")
# # 输出: user2: User(username='Bob', age=25, tags=['developer', 'pythonist'])
# #
# print("\n--- 测试 3: 验证可变默认值的独立性 ---")
# # 这是使用 field(default_factory=list) 的关键原因
# # user3 = User("lisi")
# # user4 = User("wangwu")
# # #
# # # # 修改 user3 的 tags 列表
# # user3.tags.append("123")
# # #
# # print(f"user3.tags: {user3.tags}")# lisi 18 [123]
# # print(f"user4.tags: {user4.tags}")# wangwu 18 []
#
# # 如果这里使用的是 tags: list = []，那么 user4.tags 也会包含 'new_tag'
# # 但因为使用了 default_factory，每个实例都有自己独立的列表
# # 输出:
# # user3.tags: ['new_tag']
# # user4.tags: []
#
#
# # class User:
# #     name: str
# #
# #
# # print("\n--- 测试 3: 验证可变默认值的独立性 ---")
# # user3 = User()
# # print(user3)
# a = list
# a = []