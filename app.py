import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import tempfile
import os
import uuid
import math

st.set_page_config(page_title="AI物理实验平台", layout="wide")
st.title("🎯 AI 物理实验平台")
st.markdown("选择实验类型，上传视频或输入参数，自动计算物理量。")

# ========== 侧边栏：实验类型选择 ==========
exp_type = st.sidebar.radio("请选择实验", ["自由落体运动", "圆周运动（俯视视频）"])

# ============================================================
# 1. 自由落体运动模块（原功能，完整保留）
# ============================================================
if exp_type == "自由落体运动":
    st.markdown("### 自由落体实验 - 测量重力加速度 g")
    st.info("上传自由落体视频，AI将自动分析。如结果异常，可尝试手动指定释放点。")

    # 侧边栏参数
    st.sidebar.header("自由落体参数")
    known_distance_mm = st.sidebar.number_input("标定线实际距离 (mm)", value=200, step=10)
    auto_calibrate = st.sidebar.checkbox("自动标定（背景有两条水平线）", value=True)
    if not auto_calibrate:
        pixel_dist_input = st.sidebar.number_input("两条标定线的像素间隔", value=400, step=10)

    manual_release_y = st.sidebar.number_input("手动释放点 y 坐标（像素，-1=自动检测）", value=-1, step=1)

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

    # 自由落体分析函数
    def analyze_freefall(video_path, fps_override, known_mm, auto_cal, manual_release_y,
                         lower_hsv, upper_hsv, use_drag, pixel_dist_input=None):
        cap = cv2.VideoCapture(video_path)
        fps = fps_override if fps_override > 0 else cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        st.info(f"视频帧率: {fps:.2f} fps")

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
        st.info(f"自动检测释放帧索引: {release_idx}")

        # 释放点 y 坐标
        if manual_release_y > 0:
            release_y = manual_release_y
            st.info(f"手动指定释放点 y = {release_y} 像素")
        else:
            pre_y = [y for idx, y in positions if idx < release_idx and idx >= release_idx-5]
            release_y = np.mean(pre_y) if pre_y else positions[0][1]
            st.info(f"自动检测释放点 y = {release_y:.1f} 像素")

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

        # 绘图（英文标签）
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

    # 自由落体主界面
    uploaded_file = st.file_uploader("📤 上传自由落体视频 (mp4, mov, avi)", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        temp_filename = f"temp_{uuid.uuid4().hex}.mp4"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.read())
        video_path = temp_filename
        st.video(video_path)

        if st.button("🚀 开始分析"):
            with st.spinner("AI 正在分析视频..."):
                px_dist = None if auto_calibrate else pixel_dist_input
                result = analyze_freefall(video_path, fps_input, known_distance_mm, auto_calibrate,
                                          manual_release_y, lower_hsv, upper_hsv, use_drag, px_dist)
            if result is None:
                st.error("分析失败，请调整参数或检查视频。")
            else:
                st.balloons()
        if os.path.exists(video_path):
            os.unlink(video_path)
    else:
        st.info("请上传一个自由落体视频开始实验。")

# ============================================================
# 2. 圆周运动模块（俯视视频）
# ============================================================
elif exp_type == "圆周运动（俯视视频）":
    st.markdown("### 圆周运动实验 - 角速度 / 线速度测量")
    st.info("上传从正上方俯视拍摄的圆周运动视频，AI将自动追踪小球，拟合圆轨迹并计算运动参数。")

    # 侧边栏参数
    st.sidebar.header("圆周运动参数")
    known_distance_mm = st.sidebar.number_input("标定参考距离 (mm)", value=100, step=10, help="例如轨道上两个标记点的实际距离")
    manual_pixel_dist = st.sidebar.number_input("参考距离对应的像素间隔", value=200, step=10, help="在视频中测量两标记点的像素距离")

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

    # 圆周运动分析函数
    def analyze_circular(video_path, fps_override, known_mm, manual_pixel_dist, lower_hsv, upper_hsv):
        cap = cv2.VideoCapture(video_path)
        fps = fps_override if fps_override > 0 else cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        st.info(f"圆周运动视频帧率: {fps:.2f} fps")

        # 检测小球位置（所有帧）
        positions = []  # (frame_index, x, y)
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
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    positions.append((frame_idx, cx, cy))
            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))
        cap.release()
        progress_bar.empty()

        if len(positions) < 10:
            st.error(f"检测到的小球点数不足10个（实际{len(positions)}），请调整颜色阈值或检查视频。")
            return None

        # 标定：像素/毫米
        pixel_per_mm = manual_pixel_dist / known_mm
        st.success(f"手动标定：像素/毫米 = {pixel_per_mm:.3f}")

        # 最小二乘圆拟合
        pts = np.array([(x, y) for _, x, y in positions])
        def fit_circle(pts):
            x = pts[:,0]
            y = pts[:,1]
            A = np.vstack([x, y, np.ones(len(x))]).T
            b = -(x**2 + y**2)
            coeff, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            A_c, B_c, C_c = coeff
            xc = -A_c / 2
            yc = -B_c / 2
            R = np.sqrt(xc**2 + yc**2 - C_c)
            return xc, yc, R
        xc, yc, R_pixel = fit_circle(pts)
        R_m = R_pixel / pixel_per_mm / 1000.0  # 转换为米
        st.info(f"拟合圆半径: {R_pixel:.1f} 像素 = {R_m:.4f} m")

        # 计算角度序列
        angles = []
        times = []
        for idx, x, y in positions:
            dx = x - xc
            dy = y - yc
            angle = np.arctan2(dy, dx)
            angles.append(angle)
            times.append(idx / fps)

        # 相位展开并线性拟合求角速度
        angles_unwrap = np.unwrap(angles)
        slope, intercept, r_value, p_value, std_err = linregress(times, angles_unwrap)
        omega = slope  # rad/s
        T = 2 * np.pi / abs(omega) if omega != 0 else 0
        v = omega * R_m
        a = omega**2 * R_m

        # 绘图1：轨迹与拟合圆
        fig, ax1 = plt.subplots(figsize=(6,6))
        ax1.scatter(pts[:,0], pts[:,1], s=5, label='Detected points')
        circle = plt.Circle((xc, yc), R_pixel, fill=False, color='r', linestyle='--', linewidth=2)
        ax1.add_patch(circle)
        ax1.set_aspect('equal')
        ax1.set_xlabel('x (pixel)')
        ax1.set_ylabel('y (pixel)')
        ax1.set_title('Trajectory and fitted circle')
        ax1.legend()
        st.pyplot(fig)

        # 绘图2：角度-时间及拟合直线
        fig2, ax2 = plt.subplots(figsize=(10,5))
        ax2.plot(times, angles_unwrap, 'b-', label='Unwrapped angle')
        ax2.plot(times, slope*np.array(times) + intercept, 'r--', label=f'Linear fit, ω = {omega:.3f} rad/s')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angle (rad)')
        ax2.set_title('Angle vs Time')
        ax2.legend()
        ax2.grid()
        st.pyplot(fig2)

        # 输出结果
        st.subheader("圆周运动测量结果")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("角速度 ω", f"{omega:.3f} rad/s")
        col2.metric("线速度 v", f"{v:.3f} m/s")
        col3.metric("周期 T", f"{T:.3f} s")
        col4.metric("向心加速度 a", f"{a:.3f} m/s²")
        st.info(f"拟合圆半径: {R_m:.4f} m")
        return omega, v, T, a, R_m

    # 圆周运动主界面
    uploaded_file = st.file_uploader("📤 上传圆周运动视频 (mp4, mov, avi)", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        temp_filename = f"temp_{uuid.uuid4().hex}.mp4"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.read())
        video_path = temp_filename
        st.video(video_path)

        if st.button("🚀 开始分析"):
            with st.spinner("AI 正在分析圆周运动..."):
                result = analyze_circular(video_path, fps_input, known_distance_mm, manual_pixel_dist, lower_hsv, upper_hsv)
            if result is None:
                st.error("分析失败，请调整参数或检查视频。")
            else:
                st.balloons()
        if os.path.exists(video_path):
            os.unlink(video_path)
    else:
        st.info("请上传一个俯视圆周运动视频，视频中需有已知距离的参考标记（如轨道两点），以便标定。")
