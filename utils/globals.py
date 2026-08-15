from threading import Event
from utils.config import setup_config, save_config
from utils.data import setup_data, save_data
from utils.hotkeys import setup_hotkeys,apply_hotkeys
from utils.smoothing import setup_smoothing
from utils.hand_sender import setup_controller
from tracker.controller.controller import setup_gestures
import cv2
from copy import deepcopy

config=setup_config()
data,default_data = setup_data()
latest_data = [0.0] * (64 + 6 + 12 + 10 + 12 + 10 + 3 + 2 + 10)
# 数据版本号：hand/face worker 每写完一帧 latest_data 自增，
# 平滑循环据此跳过无新数据的轮次（120Hz 循环 → 实际只按 30fps 数据源处理，数学等价）
latest_data_version = 0
current_fps = 30
stop_event = Event()
controller=setup_controller()
hotkey_config = setup_hotkeys()
smoothing_config = setup_smoothing()
gesture_config = setup_gestures()
face_landmarks=None
hand_landmarks=None
handedness=None

tongue_model=None
face_detector=None
hand_detector=None
hand_feature_model=None
hand_regression_model=None
start_time=cv2.getTickCount()

def update_configs():
    global config, data,default_data,latest_data,head_pos,controller,hotkey_config,smoothing_config,gesture_config
    config = setup_config()
    data,default_data=setup_data()
    latest_data = [0.0] * (64 + 6 + 12 + 10 + 12 + 10 + 3 + 2 + 10)
    controller=setup_controller()
    hotkey_config=setup_hotkeys()
    smoothing_config=setup_smoothing()
    gesture_config=setup_gestures()
    apply_hotkeys()

def save_configs():
    global config, data, default_data
    save_config(config)
    for idx,d in enumerate(data["BlendShapes"]):
        default_data["BlendShapes"][idx]["s"]=d["s"]
    save_data(default_data)
