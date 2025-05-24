import pandas as pd
import os
import time


time_start = time.time()
# 初始化一个空的DataFrame列表，用于存储每个文件的数据
dfs = []

# 遍历所有的CSV文件
seg_csv_files = [os.path.join('./', f) for f in os.listdir('./') if f.startswith("segment") and f.endswith('.csv')]

for file_name in seg_csv_files:
    print("reading: ", file_name)
    # 读取CSV文件并添加到列表中
    dfs.append(pd.read_csv(file_name))

# 使用concat函数合并所有的DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

print(combined_df.shape)

# 保存合并后的DataFrame到一个新的CSV文件
combined_df.to_csv('uvwMap1M_450day_half.csv', index=False)

time_end = time.time()
print("cost time: ", time_end - time_start)   # 


