import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tempfile
import os
import uuid

# ========== 修复中文字体 ==========
import matplotlib
# 尝试使用 Windows 常见中文字体
font_candidates = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
for font in font_candidates:
    try:
        matplotlib.rcParams['font.sans-serif'] = [font]
        matplotlib.rcParams['axes.unicode_minus'] = False
        # 测试该字体是否可用
        plt.text(0.5, 0.5, '测试', fontsize=10)
        plt.close()
        break
    except:
        continue
# =================================

st.set_page_config(page_title="AI自由落体实验", layout="wide")
st.title("🎯 AI 自由落体实验 - 重力加速度测量")
st.markdown("上传一个自由落体视频，AI将自动分析并计算重力加速度 g。")

# 侧边栏参数
st.sidebar.header("参数设置")
known_distance_mm = st.sidebar.number_input("标定线实际距离 (mm)", value=200, step=10)
auto_calibrate = st.sidebar.checkbox("自动标定（背景有两条水平线）", value=True)
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

# 物理模型
def ideal_model(t, g):
    return 0.5 * g * t**2

def drag_model(t, g, k):
    return (g/k)*t - (g/k**2)*(1 - np.exp(-k*t))

def analyze_video(video_path, fps_override, known_mm, auto_cal, lower_hsv, upper_hsv, use_drag):
    cap = cv2.VideoCapture(video_path)
    fps = fps_override if fps_override > 0 else cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    st.info(f"视频帧率: {fps:.2f} fps")

    positions = []
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
        st.error("检测到的小球点数不足10个，请检查视频或调整颜色阈值。")
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
        pixel_dist_input = st.sidebar.number_input("两条标定线的像素间隔", value=400, step=10)
        pixel_per_mm = pixel_dist_input / known_mm

    # 释放帧检测
    release_idx = 0
    for i in range(1, len(positions)):
        dt = (positions[i][0] - positions[i-1][0]) / fps
        if dt == 0:
            continue
        v = (positions[i][1] - positions[i-1][1]) / dt
        if v < -50:
            release_idx = positions[i][0]
            break
    if release_idx == 0:
        release_idx = positions[0][0]
    st.info(f"释放帧索引: {release_idx}")

    pre_y = [y for idx, y in positions if idx < release_idx and idx >= release_idx-5]
    release_y = np.mean(pre_y) if pre_y else positions[0][1]
    st.write(f"释放点 y 坐标: {release_y:.1f} 像素")

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
        st.error("有效轨迹点不足5个")
        return None

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

    # ========== 绘图（再次确保中文字体）==========
    plt.rcParams['font.sans-serif'] = matplotlib.rcParams['font.sans-serif']  # 沿用之前设置
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(times, hs, s=10, label='实测数据', color='blue')
    t_smooth = np.linspace(0, max(times), 100)
    if use_drag:
        h_smooth = drag_model(t_smooth, g, k)
        label = f'拟合 (含阻力) g={g:.2f}'
    else:
        h_smooth = ideal_model(t_smooth, g)
        label = f'拟合 (理想) g={g:.2f}'
    ax.plot(t_smooth, h_smooth, 'r-', label=label)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Falling Height (m)')
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

# 主界面
uploaded_file = st.file_uploader("📤 上传视频文件 (支持 mp4, mov, avi)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    temp_filename = f"temp_{uuid.uuid4().hex}.mp4"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.read())
    video_path = temp_filename
    st.video(video_path)

    if st.button("🚀 开始分析"):
        with st.spinner("AI 正在分析视频，请稍候..."):
            result = analyze_video(video_path, fps_input, known_distance_mm, auto_calibrate, lower_hsv, upper_hsv, use_drag)
        if result is None:
            st.error("分析失败，请调整参数或检查视频质量。")
        else:
            st.balloons()

    if os.path.exists(video_path):
        os.unlink(video_path)
else:
    st.info("请上传一个自由落体视频开始实验。")
