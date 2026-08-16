# -*- coding: utf-8 -*-
"""长跑监控：真实 Tracker（推理+平滑+发送）连续运行，监控 RSS/CPU/帧处理耗时。
用法: python longrun_probe.py [duration_seconds]
"""
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import psutil
import numpy as np
import utils.globals as g
from utils.tracking import Tracker

DURATION = int(sys.argv[1]) if len(sys.argv) > 1 else 300
proc = psutil.Process()

# 输入源：默认合成帧（隔离摄像头链路）；传 camera 参数用虚拟摄像头
use_camera = len(sys.argv) > 2 and sys.argv[2] == "camera"
cap = None
if use_camera:
    cap = cv2.VideoCapture(1400, cv2.CAP_ANY)
    if not cap.isOpened():
        print("BestCam 1400 打开失败，回退合成帧")
        cap = None
    else:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

noise = np.random.default_rng(42)

tracker = Tracker()
print("Tracker 就绪，开始长跑", DURATION, "s")

interval_sums = []
t0 = time.perf_counter()
last_report = t0
frames = 0

while time.perf_counter() - t0 < DURATION:
    if cap is not None:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        rgb = cv2.cvtColor(cv2.resize(frame, (800, 450)), cv2.COLOR_BGR2RGB)
    else:
        rgb = noise.integers(0, 256, (450, 800, 3), dtype=np.uint8)
    t_f = time.perf_counter()
    tracker.process_frame(rgb)
    interval_sums.append(time.perf_counter() - t_f)
    frames += 1

    now = time.perf_counter()
    if now - last_report >= 10:
        mem = proc.memory_info().rss / 1e6
        cpu = proc.cpu_percent(interval=None)
        avg_ms = sum(interval_sums) / len(interval_sums) * 1000
        print(f"t={now - t0:5.0f}s RSS={mem:7.1f}MB CPU={cpu:5.1f}% process={avg_ms:.2f}ms 帧={frames}")
        interval_sums = []
        last_report = now

cap.release()
tracker.stop()
print("完成")
