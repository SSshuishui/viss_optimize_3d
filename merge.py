import numpy as np

# 初始化一个空的数组，用于存储所有文件的和
total_sum = None

# 循环读取每个文件，并累加它们的内容
for i in range(1, 28):  # 从1到450
    filename = f"result/C{i}day1M.txt"  # 构造文件名
    # 读取文件内容
    data = np.loadtxt(filename)
    # 如果total_sum是None，初始化它
    if total_sum is None:
        total_sum = data
    else:
        # 累加数据
        total_sum += data

# 将累加的结果保存到新的文件C.txt中
np.savetxt("C.txt", total_sum, fmt='%.6f')  # 使用科学记数法格式化输出