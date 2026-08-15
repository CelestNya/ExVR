# -*- coding: utf-8 -*-
"""
ExVR 独立全管线 benchmark：多处打点采集（不改源码，全部 monkeypatch）。

覆盖的真实链路（与 main.py 一致）：
  ReadThread: cap.read → crop+resize → cvtColor BGR2RGB → (flip) → Tracker.process_frame
  HandWorker / FaceWorker: LatestFrameWorker（最新帧覆盖）
  OrtRunScheduler: 所有 DML session.run 经单线程队列（HAND 优先）
  SmoothThread: apply_smoothing 120Hz
  SendThread: data_send_thread 60Hz (UDP 127.0.0.1)

打点位置：
  采集: read_wait / crop_resize / cvtcolor / flip / submit
  推理: hand_worker_total / hand_detect / hand_landmark / face_worker_total /
        face_detect_rect / face_transform(SVD) / tongue_roi / tongue_detect
  后处理: hand_pred_handling / face_pred_handling
  ORT: 每个 session 的 排队等待(queue) + 执行(run) 分离计时
  线程: psutil 线程级 CPU 占比（含 OrtRunScheduler）
  帧率/覆盖: read/hand/face 帧数、hand/face 丢帧比、检测命中率

用法: python bench_profile.py <camera_index> <seconds>
"""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import psutil

CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 1400
SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 30

# ---------------------------------------------------------------------------
# 计时统计器
# ---------------------------------------------------------------------------
class Stats:
    def __init__(self):
        self.data = {}

    def add(self, name, ms):
        self.data.setdefault(name, []).append(ms)

    def report(self):
        rows = []
        for name, times in self.data.items():
            if not times:
                continue
            arr = np.asarray(times)
            rows.append((name, len(arr), arr.mean(), np.median(arr),
                         np.percentile(arr, 95), arr.max()))
        rows.sort(key=lambda r: -r[2])
        print(f"\n{'阶段':<24}{'次数':>7}{'均值ms':>9}{'中位ms':>9}{'p95ms':>9}{'maxms':>9}")
        print("-" * 70)
        for name, n, mean, med, p95, mx in rows:
            print(f"{name:<24}{n:>7}{mean:>9.3f}{med:>9.3f}{p95:>9.3f}{mx:>9.3f}")
        return rows


STATS = Stats()

# ---------------------------------------------------------------------------
# 1. 替换 OrtRunScheduler：排队等待 与 session.run 执行 分离计时
# ---------------------------------------------------------------------------
import utils.ort_scheduler as ort_sched

SESSION_NAMES = {}  # id(session) -> 显示名（模型初始化后填写）


class TimedScheduler(ort_sched._OrtRunScheduler):
    def run(self, session, output_names, input_feed, priority):
        t0 = time.perf_counter_ns()
        result = super().run(session, output_names, input_feed, priority)
        STATS.add(f"ort[{SESSION_NAMES.get(id(session), '?')}] total(排队+执行)", (time.perf_counter_ns() - t0) / 1e6)
        return result

    def _worker_loop(self):
        while True:
            _, _, request = self._queue.get()
            name = SESSION_NAMES.get(id(request["session"]), "?")
            t0 = time.perf_counter_ns()
            try:
                request["result"] = request["session"].run(
                    request["output_names"], request["input_feed"]
                )
            except Exception as exc:
                request["error"] = exc
            finally:
                STATS.add(f"ort[{name}] run", (time.perf_counter_ns() - t0) / 1e6)
                request["done"].set()
                self._queue.task_done()


ort_sched._ORT_RUN_SCHEDULER = TimedScheduler()

# ---------------------------------------------------------------------------
# 2. monkeypatch 各阶段（在 import utils.tracking / 初始化模型 之前）
# ---------------------------------------------------------------------------
import tracker.hand.hand as hand_mod
import tracker.face.face as face_mod
import tracker.face.tongue as tongue_mod
import tracker.face.directml_face as dmlface_mod
import tracker.hand.directml_hands as dmlhands_mod


def _wrap(name, fn):
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            STATS.add(name, (time.perf_counter_ns() - t0) / 1e6)
    return wrapped


# 后处理 / 回调
hand_mod.hand_pred_handling = _wrap("hand_pred_handling", hand_mod.hand_pred_handling)
face_mod.face_pred_handling = _wrap("face_pred_handling", face_mod.face_pred_handling)
# tongue（face.py 内绑定 + tongue.py 模块两处都换）
face_mod.mouth_roi_on_image = _wrap("tongue_roi_crop", face_mod.mouth_roi_on_image)
face_mod.detect_tongue = _wrap("tongue_detect", face_mod.detect_tongue)
tongue_mod.mouth_roi_on_image = _wrap("tongue_roi_crop", tongue_mod.mouth_roi_on_image)
tongue_mod.detect_tongue = _wrap("tongue_detect", tongue_mod.detect_tongue)

# 推理类方法
dmlhands_mod.DirectMLHands.process_frame = _wrap("hand_worker_total", dmlhands_mod.DirectMLHands.process_frame)
dmlhands_mod.DirectMLHands._detect = _wrap("hand_detect", dmlhands_mod.DirectMLHands._detect)
dmlhands_mod.DirectMLHands._run_landmark = _wrap("hand_landmark_crop+run", dmlhands_mod.DirectMLHands._run_landmark)
dmlface_mod.DirectMLFaceLandmarker.process_frame = _wrap("face_worker_total", dmlface_mod.DirectMLFaceLandmarker.process_frame)
dmlface_mod.DirectMLFaceLandmarker._detect_rect = _wrap("face_detect_rect", dmlface_mod.DirectMLFaceLandmarker._detect_rect)
dmlface_mod.DirectMLFaceLandmarker.detect = _wrap("face_detect_total", dmlface_mod.DirectMLFaceLandmarker.detect)
dmlface_mod._estimate_transform = _wrap("face_transform(SVD)", dmlface_mod._estimate_transform)

# hand_pred_handling 的绑定点在 utils.tracking，须在 import 之后才生效
import utils.tracking
from utils.tracking import Tracker
import utils.globals as g
from tracker.face.tongue import initialize_tongue_model
from tracker.face.face import initialize_face
from tracker.hand.hand import initialize_hand, initialize_hand_depth

# ---------------------------------------------------------------------------
# 3. 模型初始化 + session 命名
# ---------------------------------------------------------------------------
print("初始化模型...")
t0 = time.perf_counter()
g.tongue_model = initialize_tongue_model()
g.face_detector = initialize_face(g.tongue_model)
g.hand_detector = initialize_hand()
g.hand_feature_model, g.hand_regression_model = initialize_hand_depth()
print(f"模型初始化耗时 {time.perf_counter() - t0:.2f}s")

SESSION_NAMES.update({
    id(g.hand_detector.detector): "hand_detector(192)",
    id(g.hand_detector.landmark): "hand_landmark(224)",
    id(g.face_detector.detector): "face_detector(128)",
    id(g.face_detector.landmark): "face_landmark(256)",
    id(g.face_detector.blendshape): "face_blendshape",
    id(g.tongue_model.session): "tongue(32)",
})

# ---------------------------------------------------------------------------
# 4. 摄像头 + 主线程模拟 VideoCaptureThread.run
# ---------------------------------------------------------------------------
threading.current_thread().name = "ReadThread"

W, H = int(g.config["Setting"]["camera_width"]), int(g.config["Setting"]["camera_height"])
cap = cv2.VideoCapture(CAM, cv2.CAP_ANY)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
g.current_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"摄像头实际 {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"
      f"@{g.current_fps:.0f}fps | 处理分辨率 {W}x{H} | provider {g.config['Model']['provider']}")


def resize_for_processing(rgb_image, tw, th):
    ih, iw = rgb_image.shape[:2]
    if iw <= 0 or ih <= 0:
        return rgb_image
    ratio = tw / th
    iratio = iw / ih
    if abs(iratio - ratio) > 0.01:
        if iratio > ratio:
            cw = int(round(ih * ratio)); x0 = max(0, (iw - cw) // 2)
            rgb_image = rgb_image[:, x0:x0 + cw]
        else:
            ch = int(round(iw / ratio)); y0 = max(0, (ih - ch) // 2)
            rgb_image = rgb_image[y0:y0 + ch, :]
    if rgb_image.shape[1] != tw or rgb_image.shape[0] != th:
        rgb_image = cv2.resize(rgb_image, (tw, th), interpolation=cv2.INTER_AREA)
    return rgb_image


# ---------------------------------------------------------------------------
# 5. 启动真实线程（worker/smooth/send）
# ---------------------------------------------------------------------------
tracker = Tracker()
stop_read = threading.Event()
intervals = []


def read_loop():
    last = time.perf_counter()
    while not stop_read.is_set():
        t_read0 = time.perf_counter_ns()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        now = time.perf_counter_ns()
        intervals.append((now - last) / 1e6)
        last = now
        STATS.add("read_wait", (now - t_read0) / 1e6)

        t0 = time.perf_counter_ns()
        rgb = resize_for_processing(frame, W, H)
        t1 = time.perf_counter_ns()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        t2 = time.perf_counter_ns()
        if g.config["Setting"]["flip_x"]:
            rgb = cv2.flip(rgb, 1)
        if g.config["Setting"]["flip_y"]:
            rgb = cv2.flip(rgb, 0)
        t3 = time.perf_counter_ns()
        tracker.process_frame(rgb)
        STATS.add("submit", (time.perf_counter_ns() - t3) / 1e6)
        STATS.add("crop_resize", (t1 - t0) / 1e6)
        STATS.add("cvtcolor", (t2 - t1) / 1e6)
        STATS.add("flip", (t3 - t2) / 1e6)


th = threading.Thread(target=read_loop, name="ReadLoop", daemon=True)
th.start()

# 预热 3 秒（模型 warmup + 调度器稳态）
time.sleep(3)
print(f"预热完成，正式采样 {SECONDS} 秒...")

# 线程 CPU 采样（每个采样点同时快照线程名，避免线程退出后无法映射）
proc = psutil.Process()
samples = []
name_snapshot = {}
end = time.time() + SECONDS
while time.time() < end:
    try:
        pts = proc.threads()
        name_snapshot.update({t.ident: t.name for t in threading.enumerate()})
        tc = {t.id: t.user_time + t.system_time for t in pts}
        samples.append(tc)
    except Exception:
        pass
    time.sleep(0.5)

stop_read.set()
tracker.stop()
th.join(timeout=3)
cap.release()

# ---------------------------------------------------------------------------
# 6. 汇总输出
# ---------------------------------------------------------------------------
wall = time.time() - (end - SECONDS)
totals = {}
for i in range(1, len(samples)):
    prev, curr = samples[i - 1], samples[i]
    for tid, v in curr.items():
        if tid in prev:
            totals[tid] = totals.get(tid, 0) + v - prev[tid]

thread_map = {t.ident: t.name for t in threading.enumerate()}
thread_map.update(name_snapshot)
named = {}
other = 0.0
for tid, secs in totals.items():
    name = thread_map.get(tid, f"tid{tid}")
    named.setdefault(name, 0.0)
    named[name] += secs

print("\n" + "=" * 70)
print(f"ExVR 全管线打点结果: {SECONDS}s | cam={CAM} | 处理 {W}x{H}")
print(f"线程 CPU 占比 (单核%):")
for name, secs in sorted(named.items(), key=lambda x: -x[1]):
    print(f"  {name:<22}{secs / wall * 100:>8.1f}%")
total_cpu = sum(named.values()) / wall * 100
print(f"  {'合计':<22}{total_cpu:>8.1f}% 单核 ≈ {total_cpu / 16:.1f}% 总CPU(16线程)")

n_read = len(intervals)
n_hand = len(STATS.data.get("hand_worker_total", []))
n_face = len(STATS.data.get("face_worker_total", []))
arr = np.asarray(intervals)
print(f"\n帧率: read {n_read / wall:.1f} fps | 帧间隔 mean {arr.mean():.1f}ms p95 {np.percentile(arr, 95):.1f}ms"
      f" | hand处理 {n_hand / wall:.1f}/s (覆盖 {n_hand / max(n_read, 1) * 100:.0f}%)"
      f" | face处理 {n_face / wall:.1f}/s (覆盖 {n_face / max(n_read, 1) * 100:.0f}%)")

n_hdet = len(STATS.data.get("hand_detect", []))
n_fdet = len(STATS.data.get("face_detect_rect", []))
print(f"detector 触发: hand_detect {n_hdet} 次 (每 {n_hand / max(n_hdet, 1):.1f} 帧),"
      f" face_detect {n_fdet} 次 (每 {n_face / max(n_fdet, 1):.1f} 帧)")

STATS.report()

# ORT 排队 vs 执行 汇总
print("\nORT 调度器视角（单线程队列，所有推理串行）:")
ort_rows = []
for name, times in STATS.data.items():
    if not name.startswith("ort["):
        continue
    kind = "total(排队+执行)" if "total" in name else "run"
    base = name.replace("ort[", "").replace("]", "").replace(" total(排队+执行)", "").replace(" run", "")
    ort_rows.append((base, kind, len(times), sum(times) / len(times)))
bases = sorted(set(r[0] for r in ort_rows))
print(f"{'session':<24}{'调用':>7}{'排队+执行均值':>14}{'run均值':>10}{'排队占比':>10}")
for base in bases:
    tot = [r for r in ort_rows if r[0] == base and r[1] == "total(排队+执行)"]
    run = [r for r in ort_rows if r[0] == base and r[1] == "run"]
    n = tot[0][2] if tot else 0
    t_mean = tot[0][3] if tot else 0
    r_mean = run[0][3] if run else 0
    queue_ratio = max(0.0, (t_mean - r_mean) / t_mean * 100) if t_mean else 0
    print(f"{base:<24}{n:>7}{t_mean:>14.3f}{r_mean:>10.3f}{queue_ratio:>9.1f}%")
print("=" * 70)
