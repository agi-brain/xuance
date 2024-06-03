import numpy as np

# 示例 Table 数据，假设为 3x3x3 的三维数组
Table = np.zeros((3, 3))

# 示例索引列表
indices = [1, 2]
Table[indices[0],indices[1]] = 1
print(Table)