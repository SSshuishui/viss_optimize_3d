#!/usr/bin/env python3
"""
run_planaware_450d_accumulate.py

实时监控 C{day}day{btag}.bin 输出文件，累加到一个 running_accum.bin，
累加成功后删除原始单日文件，减少 450 天长任务的磁盘占用。

典型用法：
  python3 run_planaware_450d_accumulate.py \
    --out-dir ./out10M \
    --btag 10M \
    --nside 4096 \
    --day-start 1 \
    --day-count 450 \
    --accum-file ./out10M/C_accum_10M_days1_450.bin \
    --cmd ./shared_operator_plan_stage2_balance ... --day_count=450 ...

说明：
- 默认使用 inplace 模式，磁盘占用最低：accum 文件 + 当前刚生成的 C day 文件。
- 如需更稳但多占一个 accum 临时文件，可加：--accum-mode atomic
- 脚本会检测文件大小是否等于 12*nside^2*sizeof(dtype)，并确认文件大小稳定后才处理。
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ProcessedRecord:
    day: int
    original_name: str
    bytes: int
    processed_at: str
    elapsed_s: float


class RealtimeAccumulator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.accum_file = Path(args.accum_file).resolve() if args.accum_file else self.out_dir / f"C_accum_{args.btag}_days{args.day_start}_{args.day_start + args.day_count - 1}.bin"
        self.state_file = Path(args.state_file).resolve() if args.state_file else self.out_dir / f"{self.accum_file.name}.state.json"
        self.journal_file = self.out_dir / f"{self.accum_file.name}.journal.json"
        self.lock_file = self.out_dir / f"{self.accum_file.name}.lock"
        self.log_file = Path(args.log_file).resolve() if args.log_file else self.out_dir / f"{self.accum_file.name}.accumulate.log"

        self.dtype = np.dtype(args.dtype)
        self.npix = 12 * int(args.nside) * int(args.nside)
        self.expected_bytes = self.npix * self.dtype.itemsize
        self.chunk_elems = max(1, int(args.chunk_mb * 1024 * 1024 // self.dtype.itemsize))

        self.day_min = int(args.day_start)
        self.day_max = int(args.day_start + args.day_count - 1)
        self.pattern = re.compile(rf"^C(\d+)day{re.escape(args.btag)}\.bin$")

        self.stop_event = threading.Event()
        self.queue: "queue.Queue[Tuple[int, Path]]" = queue.Queue()
        self.queued_days: Set[int] = set()
        self.processed_days: Set[int] = set()
        self.records: List[ProcessedRecord] = []
        self.failed = 0
        self.lock_fd: Optional[int] = None
        self.subproc: Optional[subprocess.Popen] = None

    # ---------- basic utilities ----------
    def now(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def log(self, msg: str) -> None:
        line = f"[{self.now()}] {msg}"
        print(line, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def acquire_lock(self) -> None:
        try:
            self.lock_fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.lock_fd, f"pid={os.getpid()} time={self.now()}\n".encode("utf-8"))
        except FileExistsError:
            raise RuntimeError(f"锁文件已存在：{self.lock_file}\n可能已有一个累加脚本在运行。如果确认没有，请手动删除该 lock 文件。")

    def release_lock(self) -> None:
        if self.lock_fd is not None:
            try:
                os.close(self.lock_fd)
            except OSError:
                pass
            self.lock_fd = None
        try:
            self.lock_file.unlink(missing_ok=True)
        except OSError:
            pass

    def atomic_write_json(self, path: Path, obj: Dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)

    # ---------- state ----------
    def load_state(self) -> None:
        if not self.state_file.exists():
            self.processed_days = set()
            self.records = []
            return
        with open(self.state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        if int(state.get("nside", self.args.nside)) != int(self.args.nside):
            raise RuntimeError("state 文件中的 nside 与当前参数不一致，请确认是否复用了错误目录。")
        if state.get("btag", self.args.btag) != self.args.btag:
            raise RuntimeError("state 文件中的 btag 与当前参数不一致，请确认是否复用了错误目录。")
        self.records = [ProcessedRecord(**r) for r in state.get("records", [])]
        self.processed_days = {r.day for r in self.records}

    def save_state(self) -> None:
        state = {
            "nside": int(self.args.nside),
            "btag": self.args.btag,
            "dtype": str(self.dtype),
            "npix": self.npix,
            "expected_bytes": self.expected_bytes,
            "day_start": self.day_min,
            "day_count": int(self.args.day_count),
            "accum_file": str(self.accum_file),
            "processed_days": sorted(self.processed_days),
            "processed_count": len(self.processed_days),
            "records": [asdict(r) for r in self.records],
            "updated_at": self.now(),
        }
        self.atomic_write_json(self.state_file, state)

    def check_leftover_transaction(self) -> None:
        # 如果上一次在累加过程中被强杀，宁可停下来人工确认，也不要静默重复累加。
        adding_files = list(self.out_dir.glob(f"C*day{self.args.btag}.bin.adding"))
        tmp_files = list(self.out_dir.glob(f"{self.accum_file.name}.tmp_add_*"))
        if self.journal_file.exists() or adding_files:
            msg = [
                "检测到上一次未完成的累加事务。为了避免重复累加，脚本不会自动继续。",
                f"journal: {self.journal_file if self.journal_file.exists() else 'None'}",
                f"adding files: {[str(p) for p in adding_files]}",
                "处理建议：确认 accumulator 是否已经包含该 day；若不确定，建议从最近备份或重新生成该 day 后再继续。",
            ]
            raise RuntimeError("\n".join(msg))
        # tmp 文件只是在正式替换前留下的临时文件，accum 主文件未改，安全删除。
        for p in tmp_files:
            self.log(f"清理残留 tmp 文件：{p.name}")
            p.unlink(missing_ok=True)

    # ---------- accumulator file ----------
    def init_accumulator(self) -> None:
        if self.accum_file.exists():
            size = self.accum_file.stat().st_size
            if size != self.expected_bytes:
                raise RuntimeError(f"累加文件大小不对：{self.accum_file} size={size}, expected={self.expected_bytes}")
            self.log(f"恢复已有累加文件：{self.accum_file} ({size / (1024**3):.2f} GiB)")
            return
        self.log(f"初始化累加文件：{self.accum_file} npix={self.npix:,}, size={self.expected_bytes / (1024**3):.2f} GiB")
        arr = np.memmap(self.accum_file, dtype=self.dtype, mode="w+", shape=(self.npix,))
        # 分块置零，避免一次性分配大内存。
        for off in range(0, self.npix, self.chunk_elems):
            end = min(off + self.chunk_elems, self.npix)
            arr[off:end] = 0
        arr.flush()
        del arr

    # ---------- file discovery ----------
    def parse_day(self, path: Path) -> Optional[int]:
        m = self.pattern.match(path.name)
        if not m:
            return None
        return int(m.group(1))

    def is_candidate_ready(self, path: Path) -> bool:
        try:
            s1 = path.stat().st_size
            if s1 != self.expected_bytes:
                # 文件还没写完时通常小于 expected；大于 expected 则需要提醒。
                if s1 > self.expected_bytes:
                    self.log(f"跳过异常大小文件：{path.name} size={s1}, expected={self.expected_bytes}")
                return False
            time.sleep(float(self.args.stable_seconds))
            s2 = path.stat().st_size
            return s1 == s2 == self.expected_bytes
        except FileNotFoundError:
            return False
        except OSError:
            return False

    def scan_once(self) -> None:
        files = sorted(self.out_dir.glob(f"C*day{self.args.btag}.bin"), key=lambda p: self.parse_day(p) or 10**18)
        for path in files:
            day = self.parse_day(path)
            if day is None:
                continue
            if day < self.day_min or day > self.day_max:
                continue
            if day in self.processed_days or day in self.queued_days:
                continue
            if not self.is_candidate_ready(path):
                continue
            self.queued_days.add(day)
            self.queue.put((day, path))
            self.log(f"发现可处理文件：day={day}, file={path.name}")

    def scanner_loop(self) -> None:
        self.log(f"扫描线程启动：out_dir={self.out_dir}")
        while not self.stop_event.is_set():
            self.scan_once()
            time.sleep(float(self.args.check_interval))

    # ---------- accumulation ----------
    def write_journal(self, day: int, source: Path, work: Path, phase: str) -> None:
        self.atomic_write_json(self.journal_file, {
            "day": day,
            "source": str(source),
            "work": str(work),
            "phase": phase,
            "time": self.now(),
            "accum_file": str(self.accum_file),
        })

    def accumulate_inplace(self, day: int, work: Path) -> None:
        acc = np.memmap(self.accum_file, dtype=self.dtype, mode="r+", shape=(self.npix,))
        src = np.memmap(work, dtype=self.dtype, mode="r", shape=(self.npix,))
        for off in range(0, self.npix, self.chunk_elems):
            end = min(off + self.chunk_elems, self.npix)
            acc[off:end] += src[off:end]
        acc.flush()
        del src
        del acc

    def accumulate_atomic(self, day: int, work: Path) -> None:
        tmp = self.out_dir / f"{self.accum_file.name}.tmp_add_C{day}"
        if tmp.exists():
            tmp.unlink()
        acc = np.memmap(self.accum_file, dtype=self.dtype, mode="r", shape=(self.npix,))
        src = np.memmap(work, dtype=self.dtype, mode="r", shape=(self.npix,))
        out = np.memmap(tmp, dtype=self.dtype, mode="w+", shape=(self.npix,))
        for off in range(0, self.npix, self.chunk_elems):
            end = min(off + self.chunk_elems, self.npix)
            out[off:end] = acc[off:end] + src[off:end]
        out.flush()
        del out
        del src
        del acc
        os.replace(tmp, self.accum_file)

    def process_file(self, day: int, path: Path) -> None:
        start = time.time()
        work = path.with_name(path.name + ".adding")
        self.log(f"开始累加 day={day}: {path.name}")

        if work.exists():
            raise RuntimeError(f"发现已有 work 文件，可能上次中断：{work}")
        os.rename(path, work)
        self.write_journal(day, path, work, "adding")

        try:
            if self.args.accum_mode == "atomic":
                self.accumulate_atomic(day, work)
            else:
                self.accumulate_inplace(day, work)

            elapsed = time.time() - start
            rec = ProcessedRecord(
                day=day,
                original_name=path.name,
                bytes=self.expected_bytes,
                processed_at=self.now(),
                elapsed_s=elapsed,
            )
            self.records.append(rec)
            self.processed_days.add(day)
            self.save_state()

            if self.args.delete_source:
                work.unlink(missing_ok=True)
                self.log(f"完成 day={day}: elapsed={elapsed:.2f}s，已删除原始文件")
            else:
                kept = path.with_name(path.name + ".processed")
                os.rename(work, kept)
                self.log(f"完成 day={day}: elapsed={elapsed:.2f}s，原始文件保留为 {kept.name}")

            self.journal_file.unlink(missing_ok=True)
        except Exception:
            self.failed += 1
            # 保留 .adding 和 journal，避免静默重复累加。
            self.log(f"day={day} 处理失败，保留 {work.name} 和 journal 供人工检查")
            raise

    def worker_loop(self) -> None:
        self.log("累加线程启动")
        while not self.stop_event.is_set():
            try:
                day, path = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self.process_file(day, path)
            except Exception as e:
                self.log(f"处理失败：day={day}, error={e}")
                # 出现累加失败时停止，避免后续状态进一步复杂。
                self.stop_event.set()
            finally:
                self.queue.task_done()

    # ---------- subprocess ----------
    def run_command(self) -> int:
        if not self.args.cmd:
            return 0
        self.log("启动主程序：" + " ".join(self.args.cmd))
        self.subproc = subprocess.Popen(
            self.args.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self.subproc.stdout is not None
        for line in self.subproc.stdout:
            print(line, end="", flush=True)
            with open(self.out_dir / "main_program.log", "a", encoding="utf-8") as f:
                f.write(line)
        rc = self.subproc.wait()
        self.log(f"主程序退出：returncode={rc}")
        return rc

    def stop_subprocess(self) -> None:
        if self.subproc is not None and self.subproc.poll() is None:
            self.log("收到停止信号，终止主程序...")
            self.subproc.terminate()
            try:
                self.subproc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.subproc.kill()

    # ---------- main ----------
    def run(self) -> int:
        self.acquire_lock()
        rc = 0
        try:
            self.load_state()
            self.check_leftover_transaction()
            self.init_accumulator()
            self.save_state()

            self.log("=" * 80)
            self.log("实时累加监控启动")
            self.log(f"out_dir={self.out_dir}")
            self.log(f"accum_file={self.accum_file}")
            self.log(f"state_file={self.state_file}")
            self.log(f"day range=[{self.day_min}, {self.day_max}], expected_bytes={self.expected_bytes}")
            self.log(f"mode={self.args.accum_mode}, delete_source={self.args.delete_source}, chunk_mb={self.args.chunk_mb}")
            self.log("=" * 80)

            def handle_signal(signum, frame):
                self.log(f"收到信号 {signum}，准备停止")
                self.stop_event.set()
                self.stop_subprocess()

            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)

            scanner = threading.Thread(target=self.scanner_loop, daemon=True)
            worker = threading.Thread(target=self.worker_loop, daemon=True)
            scanner.start()
            worker.start()

            if self.args.cmd:
                rc = self.run_command()
                # 主程序结束后继续扫描一段时间，把最后一个文件处理掉。
                deadline = time.time() + float(self.args.drain_seconds)
                while time.time() < deadline:
                    self.scan_once()
                    if self.queue.empty():
                        # 如果已经达到预期数量，可以直接结束。
                        if len(self.processed_days) >= int(self.args.day_count):
                            break
                    time.sleep(1)
            else:
                while not self.stop_event.is_set():
                    time.sleep(2)

            # 等待队列清空。
            self.queue.join()
            self.stop_event.set()

            missing = [d for d in range(self.day_min, self.day_max + 1) if d not in self.processed_days]
            self.log("=" * 80)
            self.log(f"累加结束：processed={len(self.processed_days)}, failed={self.failed}, missing_count={len(missing)}")
            if missing:
                self.log(f"未处理 day 示例：{missing[:20]}{' ...' if len(missing) > 20 else ''}")
            self.log(f"最终累加文件：{self.accum_file}")
            self.log("=" * 80)
            return rc
        finally:
            self.release_lock()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="实时累加并删除 C{day}day{btag}.bin 输出文件")
    p.add_argument("--out-dir", required=True, help="程序输出目录，例如 ./out10M")
    p.add_argument("--btag", default="10M", help="文件名中的频率标签，例如 10M")
    p.add_argument("--nside", type=int, required=True, help="HEALPix NSIDE，例如 4096")
    p.add_argument("--day-start", type=int, default=1)
    p.add_argument("--day-count", type=int, default=450)
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="C day 文件和累加文件 dtype")
    p.add_argument("--accum-file", default=None, help="累加输出文件；默认放在 out_dir 下")
    p.add_argument("--state-file", default=None, help="处理状态 JSON；默认放在 out_dir 下")
    p.add_argument("--log-file", default=None, help="累加日志；默认放在 out_dir 下")
    p.add_argument("--accum-mode", choices=["inplace", "atomic"], default="inplace", help="inplace 磁盘占用最低；atomic 更稳但多占一个临时 accum 文件")
    p.add_argument("--delete-source", action=argparse.BooleanOptionalAction, default=True, help="累加成功后删除单日 C 文件")
    p.add_argument("--check-interval", type=float, default=5.0, help="扫描间隔秒")
    p.add_argument("--stable-seconds", type=float, default=3.0, help="文件大小稳定检测秒数")
    p.add_argument("--chunk-mb", type=int, default=256, help="分块累加大小，单位 MB")
    p.add_argument("--drain-seconds", type=float, default=120.0, help="主程序结束后继续扫描等待秒数")
    p.add_argument("--cmd", nargs=argparse.REMAINDER, help="可选：要启动的主程序命令。必须放在最后。")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    daemon = RealtimeAccumulator(args)
    return daemon.run()


if __name__ == "__main__":
    raise SystemExit(main())
