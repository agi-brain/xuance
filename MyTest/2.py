from collections import defaultdict

Q = defaultdict(lambda: 0.0)

# 访问不存在的键时，自动将该键的值设为0.0
print(Q[(1, 2, 3),0])  # 输出: 0.0
print(Q)  # 输出: defaultdict(<function <lambda> at ...>, {(1, 2, 3, 0): 0.0})
