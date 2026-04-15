import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tempfile
import os
import uuid

st.set_page_config(page_title="AI自由落体实验", layout="wide")
st.title("🎯 AI 自由落体实验 - 重力加速度测量")
st.markdown("上传自由落体视频，AI自动分析。如结果异常，可尝试手动指定释放点。")

# ========== 侧边栏参数 ==========
st.sidebar.header("参数设置")
known_distance_mm = st.sidebar.number_input("标定线实际距离 (mm)", value=200, step=10)
auto_calibrate = st.sidebar.checkbox("自动标定（背景有两条水平线）", value=True)
if not auto_calibrate:
    pixel_dist_input = st.sidebar.number_input("两条标定线的像素间隔", value=400, step=10)

# 手动释放点（-1 表示自动检测）
manual_release_y = st.sidebar.number_input("手动释放点 y 坐标（像素，-1=自动检测）", value=-1, step=1)

# 颜色阈值
manual_hsv = st.sidebar.checkbox("手动调整颜色阈值", value=False)
if manual_hsv:
    h_min = st.sidebar.slider("Hue 最小值", 0, 180, 0)
    h_max = st.sidebar.slider("Hue 最大值", 0, 180, 10)
    s_min = st.sidebar.slider("Saturation 最小值", 0, 255, 100)
    s_max = st.sidebar.slider("Saturation 最大值", 0, 255, 255)
    v_min = st.sidebar.slider("Value 最小值", 0, 255, 100)
    v_max = st.sidebar.slider("Value 最大值", 0, 255, 255)
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])
else:
    lower_hsv = np.array([0, 100, 100])
    upper_hsv = np.array([10, 255, 255])

fps_input = st.sidebar.number_input("视频帧率 (fps，0=自动检测)", value=0.0, step=1.0)
use_drag = st.sidebar.checkbox("使用空气阻力模型拟合", value=False)

# ========== 物理模型 ==========
def ideal_model(t, g):
    return 0.5 * g * t**2

def drag_model(t, g, k):
    return (g/k)*t - (g/k**2)*(1 - np.exp(-k*t))

# ========== 分析函数 ==========
def analyze_video(video_path, fps_override, known_mm, auto_cal, manual_release_y,
                  lower_hsv, upper_hsv, use_drag, pixel_dist_input=None):
    cap = cv2.VideoCapture(video_path)
    fps = fps_override if fps_override > 0 else cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    st.info(f"视频帧率: {fps:.2f} fps")

    # 检测小球位置
    positions = []  # (frame_index, y_pixel)
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cy = M["m01"] / M["m00"]
                positions.append((frame_idx, cy))
        frame_idx += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
    cap.release()
    progress_bar.empty()

    if len(positions) < 10:
        st.error(f"检测到的小球点数不足10个（实际{len(positions)}），请检查视频或调整颜色阈值。")
        return None

    # 标定
    if auto_cal:
        cap0 = cv2.VideoCapture(video_path)
        ret, frame0 = cap0.read()
        cap0.release()
        if not ret:
            st.error("无法读取视频第一帧")
            return None
        gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        hor_proj = np.sum(edges, axis=1)
        height = len(hor_proj)
        peak_thresh = np.max(hor_proj) * 0.5
        peaks = []
        for y in range(1, height-1):
            if hor_proj[y] > peak_thresh and hor_proj[y] >= hor_proj[y-1] and hor_proj[y] >= hor_proj[y+1]:
                peaks.append(y)
        if len(peaks) >= 2:
            y0 = peaks[0]
            y1 = peaks[-1]
        else:
            st.warning("自动标定失败，请使用手动标定或检查背景水平线。")
            return None
        pixel_dist = abs(y1 - y0)
        pixel_per_mm = pixel_dist / known_mm
        st.success(f"自动标定成功：像素/毫米 = {pixel_per_mm:.3f} (两条线y={y0}, {y1})")
    else:
        if pixel_dist_input is None:
            st.error("手动标定需要提供像素间隔")
            return None
        pixel_per_mm = pixel_dist_input / known_mm
        st.info(f"手动标定：像素/毫米 = {pixel_per_mm:.3f}")

    # 释放帧检测（速度突变）
    release_idx = 0
    for i in range(1, len(positions)):
        dt = (positions[i][0] - positions[i-1][0]) / fps
        if dt == 0:
            continue
        v = (positions[i][1] - positions[i-1][1]) / dt
        if v < -50:  # 速度阈值（像素/秒）
            release_idx = positions[i][0]
            break
    if release_idx == 0:
        release_idx = positions[0][0]
    st.info(f"自动检测释放帧索引: {release_idx}")

    # 释放点 y 坐标：优先使用手动输入，否则自动取释放帧前几帧平均
    if manual_release_y > 0:
        release_y = manual_release_y
        st.info(f"手动指定释放点 y = {release_y} 像素")
    else:
        pre_y = [y for idx, y in positions if idx < release_idx and idx >= release_idx-5]
        release_y = np.mean(pre_y) if pre_y else positions[0][1]
        st.info(f"自动检测释放点 y = {release_y:.1f} 像素")

    # 提取下落数据（注意：下落时 y 增加，所以 h = (y - release_y) / pixel_per_mm）
    times = []
    hs = []
    for idx, y in positions:
        if idx >= release_idx:
            t = (idx - release_idx) / fps
            h_mm = (y - release_y) / pixel_per_mm
            h_m = h_mm / 1000.0
            times.append(t)
            hs.append(h_m)

    if len(times) < 5:
        st.error(f"有效轨迹点不足5个（实际{len(times)}）")
        return None

    # 拟合
    times_arr = np.array(times)
    hs_arr = np.array(hs)
    try:
        if use_drag:
            popt, _ = curve_fit(drag_model, times_arr, hs_arr, p0=[9.8, 5])
            g, k = popt
            st.success(f"含阻力拟合：g = {g:.3f} m/s², 阻尼系数 k = {k:.3f}")
        else:
            popt, pcov = curve_fit(ideal_model, times_arr, hs_arr, p0=[9.8])
            g = popt[0]
            g_err = np.sqrt(pcov[0,0]) if pcov.shape == (1,1) else None
            st.success(f"理想模型拟合：g = {g:.3f} ± {g_err:.3f} m/s²" if g_err else f"g = {g:.3f} m/s²")
        rel_err = abs(g - 9.8)/9.8 * 100
        st.metric("相对误差", f"{rel_err:.2f}%")
    except Exception as e:
        st.error(f"拟合失败: {e}")
        return None

    # 绘图（英文标签避免乱码）
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(times, hs, s=10, label='Data', color='blue')
    t_smooth = np.linspace(0, max(times), 100)
    if use_drag:
        h_smooth = drag_model(t_smooth, g, k)
        label = f'Drag model g={g:.2f}'
    else:
        h_smooth = ideal_model(t_smooth, g)
        label = f'Ideal model g={g:.2f}'
    ax.plot(t_smooth, h_smooth, 'r-', label=label)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Height (m)')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    if not use_drag:
        h_pred = ideal_model(times_arr, g)
        residuals = (hs_arr - h_pred) * 1000
        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.scatter(times, residuals, s=10, color='green')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Residual (mm)')
        ax2.grid()
        st.pyplot(fig2)

    return g

# ========== 主界面 ==========
uploaded_file = st.file_uploader("📤 上传视频文件 (mp4, mov, avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    temp_filename = f"temp_{uuid.uuid4().hex}.mp4"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.read())
    video_path = temp_filename

    st.video(video_path)

    if st.button("🚀 开始分析"):
        with st.spinner("AI 正在分析视频..."):
            # 传递手动标定像素间隔（如果关闭自动标定）
            px_dist = None if auto_calibrate else pixel_dist_input
            result = analyze_video(video_path, fps_input, known_distance_mm, auto_calibrate,
                                   manual_release_y, lower_hsv, upper_hsv, use_drag, px_dist)
        if result is None:
            st.error("分析失败，请调整参数或检查视频。")
        else:
            st.balloons()

    if os.path.exists(video_path):
        os.unlink(video_path)
else:
    st.info("请上传一个自由落体视频开始实验。")
