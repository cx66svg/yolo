"""验证单实例锁：A 持锁时 B 必须拿不到。"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\Administrator\Desktop\livestock_posture")
PY_EXE = sys.executable
LOCK = ROOT / ".app.lock"

HOLDER = """
import sys, time
from PyQt6.QtCore import QLockFile
lock = QLockFile(r'%s'); lock.setStaleLockTime(0)
ok = lock.tryLock(100)
print('A_LOCK', ok, flush=True)
time.sleep(4)
lock.unlock()
""" % LOCK

TRYER = """
from PyQt6.QtCore import QLockFile
lock = QLockFile(r'%s'); lock.setStaleLockTime(0)
print('B_LOCK', lock.tryLock(100), flush=True)
""" % LOCK

holder = subprocess.Popen([PY_EXE, "-c", HOLDER], stdout=subprocess.PIPE, text=True)
line = holder.stdout.readline().strip()
print("第一个进程:", line)
time.sleep(0.5)

result = subprocess.run([PY_EXE, "-c", TRYER], capture_output=True, text=True, timeout=30)
second = result.stdout.strip()
print("第二个进程:", second)

holder.wait(timeout=30)

ok_a = line.endswith("True")
ok_b = second.endswith("False")
print()
print("判定：")
print(f"  A 拿到锁: {ok_a}")
print(f"  B 被拦下: {ok_b}")
print("  结论:", "通过" if ok_a and ok_b else "未通过")
sys.exit(0 if ok_a and ok_b else 1)
