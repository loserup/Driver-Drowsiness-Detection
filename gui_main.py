#!/usr/bin/env python
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
from datetime import datetime
import csv
import queue
import threading
import shutil
import os
import tkinter as tk
from tkinter import messagebox, ttk
import tkinter.font as tkFont
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords

# 初始化全局变量
data_queue = queue.Queue()
BATCH_SIZE = 300  # 每100条数据写入一次
stop_event = threading.Event()
csv_thread = None
vs = None
detection_running = False
start_time = None
timer_id = None

# 检查磁盘空间（兼容 Windows 和 POSIX 系统）
def check_disk_space(path, min_space_mb=100):
    try:
        usage = shutil.disk_usage(path)
        free_space_mb = usage.free / (1024 * 1024)  # 转换为 MB
        return free_space_mb > min_space_mb
    except Exception as e:
        print(f"[ERROR] Failed to check disk space: {e}")
        return True  # 如果检查失败，假设空间足够以继续运行

# CSV写入线程函数
def write_to_csv(filename, data_queue, stop_event):
    try:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp']
            for i in range(1, 69):
                fieldnames.extend([f'x{i}', f'y{i}'])
            fieldnames.extend(['EAR', 'MAR', 'head_tilt_degree'])
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            batch = []
            while not (stop_event.is_set() and data_queue.empty()):
                try:
                    data = data_queue.get(timeout=1.0)
                    batch.append(data)
                    if len(batch) >= BATCH_SIZE:
                        for row in batch:
                            writer.writerow(row)
                        csvfile.flush()
                        batch = []
                except queue.Empty:
                    if batch:
                        for row in batch:
                            writer.writerow(row)
                        csvfile.flush()
                        batch = []
                except Exception as e:
                    print(f"[ERROR] CSV writing error in {filename}: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to open/write {filename}: {e}")

# 主检测函数
def run_detection(subject_name, status_label, time_label, root):
    global vs, csv_thread, detection_running, start_time
    if detection_running:
        return

    # 创建被试文件夹
    subject_dir = os.path.join(os.getcwd(), subject_name)
    try:
        os.makedirs(subject_dir, exist_ok=True)
    except Exception as e:
        messagebox.showerror("错误", f"无法创建被试文件夹 {subject_dir}: {e}")
        return

    # 检查磁盘空间
    if not check_disk_space(subject_dir):
        messagebox.showerror("错误", "磁盘空间不足！请释放至少100MB空间。")
        return

    # 初始化视频流
    try:
        print("[INFO] initializing camera...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
    except Exception as e:
        messagebox.showerror("错误", f"无法初始化摄像头: {e}")
        return

    # 初始化dlib的人脸检测器和面部关键点预测器
    try:
        print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')
    except Exception as e:
        messagebox.showerror("错误", f"无法加载面部关键点预测器: {e}")
        vs.stop()
        return

    # 设置帧分辨率
    frame_width = 1024
    frame_height = 576

    # 定义关键点坐标
    image_points = np.array([
        (359, 391),  # 鼻尖 34
        (399, 561),  # 下巴 9
        (337, 297),  # 左眼左角 37
        (513, 301),  # 右眼右角 46
        (345, 465),  # 左嘴角 49
        (453, 469)   # 右嘴角 55
    ], dtype="double")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.79
    EYE_AR_CONSEC_FRAMES = 3
    COUNTER = 0

    (mStart, mEnd) = (49, 68)

    # 初始化CSV文件
    csv_filename = os.path.join(subject_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")
    global stop_event, csv_thread
    stop_event = threading.Event()
    csv_thread = threading.Thread(target=write_to_csv, args=(csv_filename, data_queue, stop_event))
    csv_thread.start()

    detection_running = True
    start_time = datetime.now()
    status_label.config(text="状态：正在检测")
    update_timer(time_label, root)

    try:
        while detection_running:
            # 读取视频帧
            frame = vs.read()
            if frame is None:
                print("[ERROR] Failed to read frame from video stream.")
                messagebox.showerror("错误", "无法读取视频帧，检测停止。")
                break
            frame = imutils.resize(frame, width=frame_width, height=frame_height)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape

            # 检测人脸
            rects = detector(gray, 0)

            if len(rects) > 0:
                text = "{} face(s) found".format(len(rects))
                cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            for rect in rects:
                # 绘制人脸边界框
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

                # 获取面部关键点
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # 计算眼睛纵横比
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # 绘制眼睛轮廓
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # 检测闭眼
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, "Eyes Closed!", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0

                # 计算嘴部纵横比
                mouth = shape[mStart:mEnd]
                mouthMAR = mouth_aspect_ratio(mouth)
                mar = mouthMAR
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 检测打哈欠
                if mar > MOUTH_AR_THRESH:
                    cv2.putText(frame, "Yawning!", (800, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 准备保存到CSV的数据
                current_time = datetime.now()
                data_row = {'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}
                for i, (x, y) in enumerate(shape, 1):
                    data_row[f'x{i}'] = x
                    data_row[f'y{i}'] = y
                data_row['EAR'] = ear
                data_row['MAR'] = mar

                # 更新头部姿态关键点并绘制
                for i, (x, y) in enumerate(shape):
                    if i in [33, 8, 36, 45, 48, 54]:
                        idx = {33: 0, 8: 1, 36: 2, 45: 3, 48: 4, 54: 5}[i]
                        image_points[idx] = np.array([x, y], dtype='double')
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                        cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                    else:
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                        cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                for p in image_points:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

                (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

                if head_tilt_degree:
                    data_row['head_tilt_degree'] = head_tilt_degree[0]
                    cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    data_row['head_tilt_degree'] = None

                # 将数据放入队列
                data_queue.put(data_row)

            # 显示帧
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # 按q键退出
            if key == ord("q"):
                break

            # 更新界面
            root.update()

    finally:
        # 清理
        stop_event.set()
        if csv_thread:
            csv_thread.join()
        if vs:
            vs.stop()
        cv2.destroyAllWindows()
        detection_running = False
        start_time = None
        time_label.config(text="检测时间：0分0秒")
        status_label.config(text="状态：已停止")

# 更新检测时间
def update_timer(time_label, root):
    global start_time, timer_id
    if detection_running and start_time:
        elapsed = datetime.now() - start_time
        minutes = elapsed.seconds // 60
        seconds = elapsed.seconds % 60
        time_label.config(text=f"检测时间：{minutes}分{seconds}秒")
        timer_id = root.after(1000, update_timer, time_label, root)
    else:
        time_label.config(text="检测时间：0分0秒")

# GUI界面
class FatigueDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("疲劳检测系统")
        self.root.geometry("400x250")
        self.root.minsize(300, 200)  # 最小窗口尺寸

        # 定义字体
        self.base_font_size = 10
        self.label_font = tkFont.Font(family="Arial", size=self.base_font_size)
        self.entry_font = tkFont.Font(family="Arial", size=self.base_font_size)
        self.button_font = tkFont.Font(family="Arial", size=self.base_font_size)

        # 容器框架
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 被试名称输入
        self.subject_label = tk.Label(self.main_frame, text="被试名称:", font=self.label_font)
        self.subject_label.pack(fill=tk.X, pady=5)
        self.subject_entry = tk.Entry(self.main_frame, font=self.entry_font)
        self.subject_entry.pack(fill=tk.X, pady=5)

        # 按钮框架
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)

        # 开始和停止按钮
        self.start_button = ttk.Button(self.button_frame, text="开始检测", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.stop_button = ttk.Button(self.button_frame, text="停止检测", command=self.stop_detection)
        self.stop_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # 状态标签
        self.status_label = tk.Label(self.main_frame, text="状态：未开始", font=self.label_font)
        self.status_label.pack(fill=tk.X, pady=5)

        # 检测时间标签
        self.time_label = tk.Label(self.main_frame, text="检测时间：0分0秒", font=self.label_font)
        self.time_label.pack(fill=tk.X, pady=5)

        # 绑定窗口调整事件
        self.root.bind("<Configure>", self.resize_fonts)

        # 窗口关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def resize_fonts(self, event):
        # 根据窗口宽度动态调整字体大小
        width = event.width
        scale = max(0.8, min(1.5, width / 400))  # 缩放比例，基于初始宽度400
        new_size = int(self.base_font_size * scale)
        self.label_font.configure(size=new_size)
        self.entry_font.configure(size=new_size)
        self.button_font.configure(size=new_size)
        # 更新按钮字体（ttk.Button 需要单独设置）
        self.start_button.configure(style="TButton")
        self.stop_button.configure(style="TButton")
        # 更新 ttk 样式
        style = ttk.Style()
        style.configure("TButton", font=self.button_font)

    def start_detection(self):
        global detection_running, start_time
        if detection_running:
            messagebox.showwarning("警告", "检测已在进行中！")
            return
        subject_name = self.subject_entry.get().strip()
        if not subject_name:
            messagebox.showerror("错误", "请输入被试名称！")
            return
        # 异步运行检测
        threading.Thread(target=run_detection, args=(subject_name, self.status_label, self.time_label, self.root), daemon=True).start()

    def stop_detection(self):
        global detection_running, timer_id
        if detection_running:
            detection_running = False
            self.status_label.config(text="状态：正在停止...")
            if timer_id:
                self.root.after_cancel(timer_id)
                timer_id = None

    def on_closing(self):
        global detection_running, timer_id
        if detection_running:
            detection_running = False
            if timer_id:
                self.root.after_cancel(timer_id)
                timer_id = None
        self.root.destroy()

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    app = FatigueDetectionApp(root)
    root.mainloop()