import pandas as pd
import os
import time

# 定义原始的uvwMap文件路径
uvw_map_path = 'uvwMap1M_450day_half.csv'

# 读取uvwMap文件，忽略第一行
uvw_map = pd.read_csv(uvw_map_path)

# 定义原始文件和更新后文件的文件夹路径
input_folder = './'
output_folder = './'

# 遍历文件
start_time = time.time()
for i in range(1, 451):
    # 构建文件名
    input_file = f'uvw{i}day1M.txt'
    output_file = f'updated_{input_file}'
    
    # 读取当前文件
    uvw_data = pd.read_csv(os.path.join(input_folder, input_file), delimiter=' ', header=None, names=['u', 'v', 'w'], skiprows=1)
    # 初始化频次为1
    uvw_data['freq'] = 1

    # 复制一份数据用于处理，不会影响原始uvw_data
    df = uvw_data.copy()
    df = df.round(2)  # 先保留两位小数，避免浮点误差
    df['u'] = (2 * df['u']).round() / 2
    df['v'] = (2 * df['v']).round() / 2
    df['w'] = (2 * df['w']).round() / 2

    # 合并数据，以u, v, w为键
    merged_data = pd.merge(df, uvw_map, on=['u', 'v', 'w'], how='left', suffixes=('', '_map'))

    # 使用map中的频次更新，如果map中没有则保持为1
    uvw_data['freq'] = merged_data['freq_map'].combine_first(uvw_data['freq'])

    # 保存更新后的数据到新文件
    uvw_data.to_csv(os.path.join(output_folder, output_file), index=False, sep=' ', header=False)

    print(f"文件 {input_file} 更新完成。")

print("所有文件已更新完成。")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time:.2f} seconds")    