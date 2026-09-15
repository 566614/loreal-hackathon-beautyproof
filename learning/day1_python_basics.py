# Day 1 作业：Python 最基础的 3 件事
# 把 Python 想成一个"听话的小助手"，你用文字吩咐它做事。

# ---------- 1) 变量：给东西起个名字，方便后面用 ----------
# 就像给一张图贴个便利贴写名字，下次直接喊名字就行
image_name = "001_real_lipstick.jpg"   # 字符串（文字）用引号包起来
score = 0.87                            # 数字（分数）不用引号

# ---------- 2) 函数：把一串动作打包，取个名，以后喊名字就执行 ----------
# 函数像"微波炉菜单"：设好"热牛奶"这个键，以后按一下就热，不用每次重设
def greet(who):                        # who 是"参数"，调用时传进来
    print("你好，" + who + "！")       # print = 让助手把字显示给你看

# ---------- 3) 读文件：让助手把文件里的字读出来给你看 ----------
# 我们读一下 data/SCHEMA.md，确认它存在（这就是后面脚本读表格的原理）
schema_path = "data/SCHEMA.md"
with open(schema_path, "r", encoding="utf-8") as f:   # open = 打开文件
    content = f.read()                                # read = 把内容读进变量

# ---------- 现在把上面学的串起来用 ----------
greet("阮")                                         # 调用函数，传"阮"进去
print("第一张图叫：" + image_name)
print("它的分数是：" + str(score))                   # str() 把数字变成文字才能拼
print("SCHEMA.md 一共有 " + str(len(content)) + " 个字")  # len() = 数一下有多少字
