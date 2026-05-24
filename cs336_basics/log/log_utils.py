import time


def log_info(msg: str):
    print(f"[BPE][{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)