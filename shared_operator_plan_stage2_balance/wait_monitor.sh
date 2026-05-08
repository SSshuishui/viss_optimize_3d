#!/bin/bash
# wait_for_gpus_and_monitor.sh

PYTHON_SCRIPT="python viss_monitor.py"  # 使用新版
CHECK_INTERVAL=30
GPUS_TO_CHECK="0,1"
MEM_THRESHOLD=500
UTIL_THRESHOLD=5

echo "[$(date)] 等待 GPU ${GPUS_TO_CHECK} 空闲..."

while true; do
    mapfile -t mem_used < <(nvidia-smi -i "$GPUS_TO_CHECK" --query-gpu=memory.used --format=csv,noheader,nounits)
    mapfile -t gpu_util < <(nvidia-smi -i "$GPUS_TO_CHECK" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    
    idle_count=0
    gpu_indices=(${GPUS_TO_CHECK//,/ })
    
    for idx in "${!gpu_indices[@]}"; do
        mem="${mem_used[$idx]// /}"
        util="${gpu_util[$idx]// /}"
        if [ "$mem" -lt "$MEM_THRESHOLD" ] && [ "$util" -lt "$UTIL_THRESHOLD" ]; then
            ((idle_count++))
        fi
    done

    if [ "$idle_count" -eq "${#gpu_indices[@]}" ]; then
        echo "[$(date)] ✅ GPU 空闲，启动任务..."
        break
    fi
    sleep "$CHECK_INTERVAL"
done

# 启动 Python 守护进程（使用 nohup 确保后台稳定运行）
echo "[$(date)] 启动文件处理守护进程 (处理速度: ~几十秒/12GB文件)..."
nohup $PYTHON_SCRIPT > viss_monitor.log 2>&1 &
MONITOR_PID=$!
echo "[$(date)] 守护进程 PID: $MONITOR_PID"

# 给守护进程5秒初始化时间
sleep 3

cleanup() {
    echo "[$(date)] 主程序结束，等待守护进程完成剩余文件处理..."
    # 等待队列清空（最多等10分钟，因为处理需要时间）
    for i in {1..60}; do
        if ! kill -0 $MONITOR_PID 2>/dev/null; then
            echo "[$(date)] 守护进程已自动退出"
            break
        fi
        # 检查队列是否为空（通过日志或标记文件）
        if [ -f "../out10M_3d_stage2_balance_absorption_split/running_accum.log" ]; then
            # 简单等待固定时间，确保最后几个文件被处理
            sleep 10
        fi
    done
    
    # 如果还在运行，强制终止
    if kill -0 $MONITOR_PID 2>/dev/null; then
        echo "[$(date)] 强制停止守护进程"
        kill $MONITOR_PID
        sleep 2
    fi
    exit 0
}

trap cleanup EXIT INT TERM
