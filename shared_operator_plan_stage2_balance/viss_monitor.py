#!/usr/bin/env python3
"""
accumulator_daemon_robust.py
健壮版文件累加守护进程 - 针对处理时间较长（几十秒）的场景优化
"""
import os
import time
import glob
import numpy as np
import re
import threading
import queue
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp

# ==================== 配置 ====================
WATCH_DIR = "../out10M_3d_stage2_balance_absorption_split/"
FIGS_DIR = "../out10M_3d_stage2_balance_absorption_split/figs"
ACCUM_FILE = os.path.join(WATCH_DIR, "running_accum.bin")
PATTERN = "C*day10M.bin"
DTYPE = np.float32
NSIDE = 4096
NEST = True

# 关键参数调整（针对处理慢的场景）
CHECK_INTERVAL = 5           # 检查间隔缩短到10秒（生成周期200秒，来得及）
FILE_STABLE_TIME = 3.0        # 确认文件写完需要稳定3秒（根据你的I/O调整）
PROCESSING_TIMEOUT = 300      # 单个文件处理超时时间（秒）
# =============================================

class AccumulatorDaemon:
    def __init__(self):
        self.processing_queue = queue.Queue()
        self.processed_files = set()
        self.accum_lock = threading.Lock()
        self.stats = {'processed': 0, 'failed': 0, 'total_size_gb': 0}
        self.running = True
        
    def ensure_dirs(self):
        os.makedirs(WATCH_DIR, exist_ok=True)
        os.makedirs(FIGS_DIR, exist_ok=True)
        
    def get_npix(self):
        return 12 * NSIDE ** 2
        
    def init_accum_file(self, npix):
        if not os.path.exists(ACCUM_FILE):
            print(f"[初始化] 创建累加文件: {npix:,} 像素 ({npix*4/1e9:.2f} GB)")
            zeros = np.zeros(npix, dtype=DTYPE)
            zeros.tofile(ACCUM_FILE)
            del zeros
        else:
            size = os.path.getsize(ACCUM_FILE)
            print(f"[恢复] 加载已有累加: {size/1e9:.2f} GB")
            
    def is_file_stable(self, filepath, wait_time=FILE_STABLE_TIME):
        """
        检查文件是否已写完（大小稳定）
        针对大文件（12GB），检查时间稍长更保险
        """
        try:
            size1 = os.path.getsize(filepath)
            if size1 == 0:
                return False
            # 对于12GB文件，等3秒确保写入完成
            time.sleep(wait_time)
            size2 = os.path.getsize(filepath)
            return size1 == size2
        except OSError:
            return False
            
    def extract_number(self, filepath):
        filename = os.path.basename(filepath)
        match = re.match(r'C(\d+)day10M\.bin', filename)
        return int(match.group(1)) if match else float('inf')
        
    def visualize(self, data, tag, title):
        """生成图片（在线程中执行避免阻塞）"""
        try:
            plt.figure(figsize=(12, 7))
            hp.mollview(data, nest=NEST, title=title, unit="K", 
                       min=np.percentile(data, 1), max=np.percentile(data, 99))
            hp.graticule()
            plt.savefig(os.path.join(FIGS_DIR, f"{tag}.png"), 
                       dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"    [画图] {tag}.png 完成")
        except Exception as e:
            print(f"    [画图错误] {e}")
            
    def process_single_file(self, filepath):
        """
        处理单个文件（耗时操作：读取12GB + 累加 + 画图 + 删除）
        预计耗时：几十秒（取决于磁盘I/O）
        """
        filename = os.path.basename(filepath)
        file_num = self.extract_number(filename)
        npix = self.get_npix()
        
        print(f"\n[{time.strftime('%H:%M:%S')}] 开始处理 #{self.stats['processed']+1}: {filename}")
        print(f"    文件大小: {os.path.getsize(filepath)/1e9:.2f} GB")
        start_time = time.time()
        
        try:
            # 1. 内存映射读取（零拷贝，但打开大文件仍需时间）
            accum = np.memmap(ACCUM_FILE, dtype=DTYPE, mode='r+', shape=(npix,))
            data = np.memmap(filepath, dtype=DTYPE, mode='r', shape=(npix,))
            
            # 2. 执行累加（CPU密集型，12GB数据）
            print(f"    累加计算中...")
            with self.accum_lock:  # 确保线程安全（虽然通常是单线程处理）
                accum[:] += data[:]
                accum.flush()
            
            # 3. 获取统计信息
            curr_max = accum.max()
            curr_min = accum.min()
            curr_mean = accum.mean()
            
            # 4. 关闭数据文件（必须先关闭才能删除）
            del data
            
            # 5. 画图（可选，较耗时，可以注释掉如果不需要实时图）
            # 注意：如果画图太耗时（超过100秒），可以移到后台线程或省略
            fig_title = f"C{file_num} | Mean:{curr_mean:.2e} Range:[{curr_min:.2e},{curr_max:.2e}]"
            self.visualize(accum, f"C{file_num}", fig_title)
            
            del accum
            
            # 6. 删除原文件
            os.remove(filepath)
            
            elapsed = time.time() - start_time
            self.stats['processed'] += 1
            self.stats['total_size_gb'] += 12  # 假设12GB
            
            print(f"    ✅ 完成 ({elapsed:.1f}s) | 累计处理: {self.stats['processed']} 个")
            print(f"    当前累加和: Mean={curr_mean:.6f}, Max={curr_max:.6f}")
            
            # 记录日志
            with open(ACCUM_FILE + ".log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | C{file_num} | "
                       f"time={elapsed:.1f}s | mean={curr_mean:.6f}\n")
                       
            return True
            
        except Exception as e:
            print(f"    ❌ 失败: {e}")
            self.stats['failed'] += 1
            return False
            
    def scanner_thread(self):
        """
        扫描线程：专门负责发现新文件，放入队列
        与处理线程分离，避免处理耗时影响检测
        """
        print(f"[扫描线程] 启动，监控: {WATCH_DIR}")
        
        while self.running:
            try:
                files = glob.glob(os.path.join(WATCH_DIR, PATTERN))
                files.sort(key=self.extract_number)
                
                for filepath in files:
                    if not self.running:
                        break
                        
                    # 跳过累加文件本身
                    if filepath == ACCUM_FILE:
                        continue
                        
                    # 跳过已处理或在队列中的
                    if filepath in self.processed_files:
                        continue
                        
                    # 检查文件是否写完（关键！对于12GB文件）
                    if not self.is_file_stable(filepath):
                        continue
                        
                    # 加入处理队列
                    print(f"[扫描] 发现新文件: {os.path.basename(filepath)}")
                    self.processing_queue.put(filepath)
                    self.processed_files.add(filepath)
                    
                time.sleep(CHECK_INTERVAL)
                
            except Exception as e:
                print(f"[扫描错误] {e}")
                time.sleep(CHECK_INTERVAL)
                
    def worker_thread(self):
        """
        工作线程：从队列取文件并处理
        如果处理速度（几十秒）< 生成周期（200秒），一个线程就够
        如果处理更慢，可以开多个worker（但注意磁盘I/O瓶颈）
        """
        print(f"[工作线程] 启动，准备处理...")
        
        while self.running:
            try:
                # 阻塞等待，但每5秒检查一次是否应该退出
                filepath = self.processing_queue.get(timeout=5)
                
                # 处理文件（耗时几十秒的操作）
                self.process_single_file(filepath)
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[工作错误] {e}")
                
    def run(self):
        print("=" * 70)
        print("累加守护进程启动 (Robust Slow-Processing Edition)")
        print("=" * 70)
        print(f"配置: 检查间隔={CHECK_INTERVAL}s, 文件稳定检测={FILE_STABLE_TIME}s")
        print(f"预计处理时间: 每文件几十秒 (读取12GB + 累加 + 画图 + 删除)")
        print("=" * 70)
        
        self.ensure_dirs()
        npix = self.get_npix()
        self.init_accum_file(npix)
        
        # 先处理目录中已存在的文件（如果有）
        print("\n[初始化] 扫描已有文件...")
        existing = glob.glob(os.path.join(WATCH_DIR, PATTERN))
        for f in sorted(existing, key=self.extract_number):
            if f != ACCUM_FILE and self.is_file_stable(f, wait_time=1.0):
                self.processing_queue.put(f)
                self.processed_files.add(f)
                print(f"  排队: {os.path.basename(f)}")
                
        # 启动线程
        scanner = threading.Thread(target=self.scanner_thread, daemon=True)
        worker = threading.Thread(target=self.worker_thread, daemon=True)
        
        scanner.start()
        worker.start()
        
        print(f"\n[运行中] 按 Ctrl+C 停止...")
        print(f"队列长度: ", end="", flush=True)
        
        try:
            while True:
                # 显示队列状态
                qsize = self.processing_queue.qsize()
                print(f"\r队列长度: {qsize} | 已处理: {self.stats['processed']} | "
                      f"已处理容量: {self.stats['total_size_gb']:.1f} GB", end="")
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n[停止] 正在关闭...")
            self.running = False
            
            # 等待当前处理完成（最多30秒）
            print("等待当前任务完成...")
            time.sleep(1)
            
            print(f"\n统计:")
            print(f"  成功处理: {self.stats['processed']} 个文件")
            print(f"  失败: {self.stats['failed']} 个")
            print(f"  总处理数据: {self.stats['total_size_gb']:.1f} GB")
            print(f"累加文件: {ACCUM_FILE}")
            print(f"日志: {ACCUM_FILE}.log")

if __name__ == "__main__":
    daemon = AccumulatorDaemon()
    daemon.run()