#!/bin/bash

# 遍历文件名的数字范围
for j in {220..450}
do
    # 构建文件名
    file="./C${j}day1M.txt"

    # 检查文件是否存在
    if [ -f "$file" ]; then
        # 统计文件的行数并打印
        lines=$(wc -l < "$file")
        echo "$file lines: $lines"
    else
        # 如果文件不存在，则提示并跳过
        echo "$file does not exist. Skipping..."
    fi
done

