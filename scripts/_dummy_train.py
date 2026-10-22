"""假训练进程：用于验证「关窗口是否真的终止了子进程」。

逐行打印模拟训练输出，持续一段时间，供生命周期测试使用。
"""
import sys
import time

LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0

start = time.time()
index = 0
while time.time() - start < LIMIT:
    index += 1
    print(f"epoch {index}/999  模拟训练中", flush=True)
    time.sleep(0.2)
