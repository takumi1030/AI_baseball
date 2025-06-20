# =============================
# app.py : スマホ投球動画解析アプリ
# 安定動作版（Colab/Streamlit Cloud対応）
# =============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt
import tempfile
import os
import io

# -----------------------------
# Streamlit 設定 & フォント設定
# -----------------------------
st.set_page_config(layout="wide")
font_path = 'NotoSansJP-Regular.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans JP'
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning(f"フォント '{font_path}' が見つかりません")

# -----------------------------
# ヘルパー関数
# -----------------------------
def normalize_curve(data_series, num_points=101):
    if len(data_series) < 2:
        return np.zeros(num_points)
    x_current = np.linspace(0, 100, len(data_series))
    x_new = np.linspace(0, 100, num_points)
    return np.interp(x_new, x_current, data_series)

def lowpass_filter(data, cutoff, fs, order=4):
    data = np.asarray(data, dtype=np.float64)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1 or len(data) <= 3 * (order + 1):
        return data
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def calc_plane_angular_velocity(p1_prev, p2_prev, p1_curr, p2_curr, dt):
    v_prev = np.array([p2_prev[0]-p1_prev[0], p2_prev[2]-p1_prev[2]])
    v_curr = np.array([p2_curr[0]-p1_curr[0], p2_curr[2]-p1_curr[2]])
    ang_prev = np.arctan2(v_prev[1], v_prev[0])
    ang_curr = np.arctan2(v_curr[1], v_curr[0])
    delta = ang_curr - ang_prev
    if delta > np.pi: delta -= 2*np.pi
    if delta < -np.pi: delta += 2*np.pi
    return np.rad2deg(delta) / dt

def calc_elbow_ext_velocity(shoulder, elbow, wrist, dt):
    v1 = shoulder - elbow
    v2 = wrist - elbow
    dot = np.einsum('ij,ij->j', v1, v2)
    norm1 = np.linalg.norm(v1, axis=0)
    norm2 = np.linalg.norm(v2, axis=0)
    cos_theta = np.clip(dot/(norm1*norm2), -1, 1)
    angles = np.arccos(cos_theta)
    deg = np.rad2deg(angles)
    return np.abs(np.diff(deg) / dt)

# -----------------------------
# UI
# -----------------------------
st.title('⚾ スマホ投球動画 運動連鎖解析ツール')

with st.sidebar:
    st.header('ステップ1: 設定')
    side = st.radio('利き腕', ('R', 'L'), format_func=lambda x: '右投手' if x=='R' else '左投手')
    cutoff = st.slider('フィルター強さ (Hz)', 4, 20, 10, 1)
    st.divider()
    st.header('ステップ2: 動画アップロード')
    uploaded = st.file_uploader('投球動画をアップロード', type=['mp4', 'avi', 'mov'])

# -----------------------------
# 動画処理
# -----------------------------
if uploaded:
    st.header('解析結果')

    # 一時ファイルに保存
    with st.spinner('動画を準備中...'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

    with st.spinner('AIが解析中... しばらくお待ちください'):
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        dt = 1/fps if fps > 0 else 1/30.0

        mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_world_landmarks:
                frames.append(results.pose_world_landmarks.landmark)
        cap.release()
        mp_pose.close()

    if len(frames) > 40:
        st.success('解析完了！データを処理中...')
        num_frames = len(frames)
        num_landmarks = len(frames[0])
        coords = np.array([[[lm.x, lm.y, lm.z] for lm in f] for f in frames]).transpose((1,2,0))
        filt_coords = np.zeros_like(coords)
        for i in range(num_landmarks):
            for j in range(3):
                filt_coords[i,j,:] = lowpass_filter(coords[i,j,:], cutoff, fps)

        stride_knee = mp.solutions.pose.PoseLandmark.LEFT_KNEE if side=="R" else mp.solutions.pose.PoseLandmark.RIGHT_KNEE
        throw_hand = mp.solutions.pose.PoseLandmark.RIGHT_INDEX if side=="R" else mp.solutions.pose.PoseLandmark.LEFT_INDEX

        start = np.argmin(filt_coords[stride_knee,1,:])
        hv = np.diff(filt_coords[throw_hand,0,:]) / dt
        release = start + np.argmax(np.abs(hv[start:])) if start < len(hv) else len(hv)

        seg = filt_coords[:,:,start:release]
        n_seg = seg.shape[2]

        idx = {
            'pelvis_l': mp.solutions.pose.PoseLandmark.LEFT_HIP,
            'pelvis_r': mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            'thorax_l': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            'thorax_r': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            'shoulder': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER if side=="R" else mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            'elbow': mp.solutions.pose.PoseLandmark.RIGHT_ELBOW if side=="R" else mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
            'wrist': mp.solutions.pose.PoseLandmark.RIGHT_WRIST if side=="R" else mp.solutions.pose.PoseLandmark.LEFT_WRIST
        }

        pelvis_v = [calc_plane_angular_velocity(seg[idx['pelvis_l'],:,i-1], seg[idx['pelvis_r'],:,i-1],
                                                seg[idx['pelvis_l'],:,i], seg[idx['pelvis_r'],:,i], dt)
                    for i in range(1, n_seg)]
        thorax_v = [calc_plane_angular_velocity(seg[idx['thorax_l'],:,i-1], seg[idx['thorax_r'],:,i-1],
                                                seg[idx['thorax_l'],:,i], seg[idx['thorax_r'],:,i], dt)
                    for i in range(1, n_seg)]
        shoulder_v = [calc_plane_angular_velocity(seg[idx['shoulder'],:,i-1], seg[idx['elbow'],:,i-1],
                                                  seg[idx['shoulder'],:,i], seg[idx['elbow'],:,i], dt)
                      for i in range(1, n_seg)]
        elbow_v = calc_elbow_ext_velocity(seg[idx['shoulder']], seg[idx['elbow']], seg[idx['wrist']], dt)

        # -----------------------------
        # プロット
        # -----------------------------
        st.subheader('運動連鎖グラフ')
        fig, ax = plt.subplots(figsize=(12, 7))
        time_norm = np.linspace(0, 100, 101)
        ax.plot(time_norm, normalize_curve(np.abs(pelvis_v)), label='骨盤 角速度', color='blue')
        ax.plot(time_norm, normalize_curve(np.abs(thorax_v)), label='胸郭 角速度', color='green')
        ax.plot(time_norm, normalize_curve(np.abs(shoulder_v)), label='肩 角速度', color='red')
        ax.plot(time_norm, normalize_curve(np.abs(elbow_v)), label='肘 伸展速度', color='purple')
        ax.set_xlabel('正規化時間 (%)')
        ax.set_ylabel('角速度 (deg/s)')
        ax.set_title('投球動作 運動連鎖')
        ax.legend(); ax.grid(True)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200)
        base = os.path.splitext(uploaded.name)[0]
        st.download_button("グラフをダウンロード", buf, f"{base}_chain.png", "image/png")
    else:
        st.error("十分な骨格データが検出できませんでした。撮影条件を確認してください。")

    # クリーンアップ
    os.remove(tmp_path)
