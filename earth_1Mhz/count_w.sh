#!/bin/bash

# 检查是否有文件存在
if ! ls uvw*.txt 1> /dev/null 2>&1; then
    echo "未找到符合条件的文件"
    exit 1
fi

# 创建临时文件存放当前文件的w列的最大和最小值
tmp_max=$(mktemp)
tmp_min=$(mktemp)

# 遍历所有符合条件的文件
for file in uvw*.txt; do
    echo "处理文件: $file"
    
    # 获取当前文件的最大w值和最小w值
    awk '{print $3}' "$file" | tee >(sort -n | head -n 1 >> "$tmp_min") | sort -n | tail -n 1 >> "$tmp_max"
done

# 计算全局最小值
global_min=$(sort -n "$tmp_min" | head -n 1)
echo "全局最小w值: $global_min"

# 计算全局最大值
global_max=$(sort -n "$tmp_max" | tail -n 1)
echo "全局最大w值: $global_max"

# 清理临时文件
rm "$tmp_max" "$tmp_min"