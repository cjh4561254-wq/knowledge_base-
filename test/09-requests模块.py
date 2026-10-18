import requests

response = requests.get("https://api.github.com/users/octocat")

print(response.json())

if response.status_code == 200:
    # 自动转换为 Python 字典
    user_info = response.json()
    print(f"用户名: {user_info['name']}")
    print(f"公司: {user_info['company']}")
else:
    print("请求失败！")