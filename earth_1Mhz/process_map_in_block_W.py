import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import re

# 定义w轴的范围和分段
w_ranges = [(-166 + i * 167, -166 + (i + 1) * 167) for i in range(2)]

def process_segment(filename, w_min, w_max):
    print("reading file: ", filename)
    # 读取txt文件并只处理属于当前w范围段的数据
    df = pd.read_csv(filename, delim_whitespace=True, header=None, names=['u', 'v', 'w'], skiprows=1)

    df = df.round(2)  # 先保留两位小数，避免浮点误差
    df['u'] = (2 * df['u']).round() / 2
    df['v'] = (2 * df['v']).round() / 2
    df['w'] = (2 * df['w']).round() / 2
    
    segment_df = df[(df['w'] >= w_min) & (df['w'] < w_max)]
    
    return segment_df

def process_and_save_segment(segment_idx, w_min, w_max, txt_files):
    # 初始化一个空DataFrame用于合并数据
    combined_segment_df = pd.DataFrame(columns=['u', 'v', 'w'])
    
    # 多线程处理文件
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 将每个文件的segment_df合并到combined_segment_df中
        for segment_df in executor.map(lambda file: process_segment(file, w_min, w_max), txt_files):
            combined_segment_df = pd.concat([combined_segment_df, segment_df], ignore_index=True)
    
    # 对合并后的数据进行一次groupby，统计所有文件中相同坐标的总频次
    combined_segment_df = combined_segment_df.groupby(['u', 'v', 'w']).size().reset_index(name='freq')

    # 筛选并保存频次大于1的部分
    freq_gt_1_df = combined_segment_df[combined_segment_df['freq'] > 1]
    if not freq_gt_1_df.empty:
        output_filename_freq_gt_1 = f"segment_{segment_idx}_freq_gt_1.csv"
        freq_gt_1_df.to_csv(output_filename_freq_gt_1, index=False)
        print(f"Segment {segment_idx} with freq > 1 saved as {output_filename_freq_gt_1}")



def main(input_dir):
    start_time = time.time()

    max_number = 450  # 可以根据需要修改
    # 获取所有符合条件的txt文件的路径
    txt_files = []
    for f in os.listdir(input_dir):
        match = re.match(r"uvw(\d+)day1M\.txt$", f)
        if match:
            number = int(match.group(1))
            if 1 <= number <= max_number:
                txt_files.append(os.path.join(input_dir, f))
    print(len(txt_files), "files found")

    # 遍历每个w轴范围段，依次处理并保存数据
    for idx, (w_min, w_max) in enumerate(w_ranges):
        print(f"Processing segment {idx}: w range {w_min} to {w_max}")
        process_and_save_segment(idx, w_min, w_max, txt_files)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")   # 1847.40s   # 4020.01s

if __name__ == "__main__":
    input_dir = "./"  # 设置为存放txt文件的目录
    main(input_dir)
