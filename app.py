# app.py (スマートフォン動画解析 専用アプリ)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import re
import os
import cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt
import matplotlib.font_manager as fm

# --- Font Setup and Page Config ---
st.set_page_config(layout="wide")
font_path = 'NotoSansJP-Regular.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans JP'
    plt.rcParams['axes.unicode_minus'] = False
else:
    st.warning(f"フォントファイル '{font_path}' が見つかりません。正しく表示されない可能性があります。")

# --- Helper Functions ---
@st.cache_data(max_entries=10) # Cache function calls for performance
def normalize_curve(data_series, num_points=101):
    if len(data_series) < 2: return np.zeros(num_points)
    current_x = np.linspace(0, 100, len(data_series))
    new_x = np.linspace(0, 100, num_points)
    return np.interp(new_x, current_x, data_series)

@st.cache_data(max_entries=10)
def lowpass_filter(data, cutoff, fs, order=4):
    data_np = np.asarray(data, dtype=np.float64)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1 or len(data_np) <= 3 * (order + 1): return data_np
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data_np)

def calculate_plane_angular_velocity(p1_prev, p2_prev, p1_curr, p2_curr, dt):
    v_prev = np.array([p2_prev[0] - p1_prev[0], p2_prev[2] - p1_prev[2]])
    v_curr = np.array([p2_curr[0] - p1_curr[0], p2_curr[2] - p1_curr[2]])
    angle_prev = np.arctan2(v_prev[1], v_prev[0]); angle_curr = np.arctan2(v_curr[1], v_curr[0])
    delta_angle = angle_curr - angle_prev
    if delta_angle > np.pi: delta_angle -= 2 * np.pi
    elif delta_angle < -np.pi: delta_angle += 2 * np.pi
    return np.rad2deg(delta_angle) / dt

def calculate_elbow_extension_velocity(shoulder_coords, elbow_coords, wrist_coords, dt):
    v1 = shoulder_coords - elbow_coords; v2 = wrist_coords - elbow_coords
    dot_product = np.einsum('ij,ij->j', v1, v2)
    norm_v1 = np.linalg.norm(v1, axis=0); norm_v2 = np.linalg.norm(v2, axis=0)
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)
    angles_deg = np.rad2deg(angles_rad)
    return np.abs(np.diff(angles_deg) / dt)

# ==============================================================================
# Main App UI and Logic
# ==============================================================================
st.title('⚾ スマートフォン動画 運動連鎖解析ツール')
st.write('スマートフォンで撮影した投球動画をアップロードするだけで、運動連鎖を全自動で解析・可視化します。')

with st.sidebar:
    st.header('ステップ1: 解析設定')
    side = st.radio('投手の利き腕を選択', ('R', 'L'), format_func=lambda x: '右投手' if x == 'R' else '左投手')
    cutoff_freq = st.slider('フィルターの強さ (Hz)', min_value=4, max_value=20, value=10, step=1, help="値を小さくするほどグラフは滑らかになります。推奨値: 6Hz～12Hz")
    
    st.divider()
    st.header('ステップ2: ファイルをアップロード')
    uploaded_file = st.file_uploader(f"解析したい{side}投手の動画ファイルを1つアップロードしてください", type=['mp4', 'mov', 'avi'])
    st.info("より良い結果のため、真横から、高フレームレート（スローモーション）で撮影した動画をお使いください。")

if uploaded_file:
    st.header('解析結果')
    
    # Save uploaded file temporarily to be read by OpenCV
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner('AIが動画を解析中... これには数分かかる場合があります。'):
        # --- Video Processing ---
        cap = cv2.VideoCapture(uploaded_file.name)
        fps = cap.get(cv2.CAP_PROP_FPS); dt = 1/fps if fps > 0 else 1/30.0
        
        mp_pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)
        all_landmarks_data = []
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            results = mp_pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.pose_world_landmarks: all_landmarks_data.append(results.pose_world_landmarks.landmark)
        cap.release(); mp_pose.close()

    if len(all_landmarks_data) > 40:
        with st.spinner('データをフィルタリング・解析中...'):
            # --- Filtering, Segmentation, and Calculation ---
            num_frames = len(all_landmarks_data)
            num_landmarks = len(all_landmarks_data[0])
            raw_coords = np.array([[[lm.x, lm.y, lm.z] for lm in frame_lms] for frame_lms in all_landmarks_data]).transpose((1,2,0))
            filtered_coords = np.zeros_like(raw_coords)
            for i in range(num_landmarks):
                for j in range(3): filtered_coords[i,j,:] = lowpass_filter(raw_coords[i,j,:], cutoff_freq, fps)

            stride_knee_idx = mp_pose.PoseLandmark.LEFT_KNEE if side == "R" else mp_pose.PoseLandmark.RIGHT_KNEE
            throwing_hand_idx = mp_pose.PoseLandmark.RIGHT_INDEX if side == "R" else mp_pose.PoseLandmark.LEFT_INDEX
            start_frame = np.argmin(filtered_coords[stride_knee_idx, 1, :])
            hand_velocity = np.diff(filtered_coords[throwing_hand_idx, 0, :]) / dt
            release_frame = start_frame + np.argmax(np.abs(hand_velocity[start_frame:])) if start_frame < len(hand_velocity) else len(hand_velocity)
            
            segmented_coords = filtered_coords[:, :, start_frame:release_frame]
            num_segmented_frames = segmented_coords.shape[2]
            
            indices = {'pelvis_l':mp_pose.PoseLandmark.LEFT_HIP,'pelvis_r':mp_pose.PoseLandmark.RIGHT_HIP,'thorax_l':mp_pose.PoseLandmark.LEFT_SHOULDER,'thorax_r':mp_pose.PoseLandmark.RIGHT_SHOULDER,'shoulder':mp_pose.PoseLandmark.RIGHT_SHOULDER if side=="R" else mp_pose.PoseLandmark.LEFT_SHOULDER,'elbow':mp_pose.PoseLandmark.RIGHT_ELBOW if side=="R" else mp_pose.PoseLandmark.LEFT_ELBOW,'wrist':mp_pose.PoseLandmark.RIGHT_WRIST if side=="R" else mp_pose.PoseLandmark.LEFT_WRIST}
            
            pelvis_vel = [calculate_plane_angular_velocity(segmented_coords[indices['pelvis_l'],:,i-1], segmented_coords[indices['pelvis_r'],:,i-1], segmented_coords[indices['pelvis_l'],:,i], segmented_coords[indices['pelvis_r'],:,i], dt) for i in range(1, num_segmented_frames)]
            thorax_vel = [calculate_plane_angular_velocity(segmented_coords[indices['thorax_l'],:,i-1], segmented_coords[indices['thorax_r'],:,i-1], segmented_coords[indices['thorax_l'],:,i], segmented_coords[indices['thorax_r'],:,i], dt) for i in range(1, num_segmented_frames)]
            shoulder_vel = [calculate_plane_angular_velocity(segmented_coords[indices['shoulder'],:,i-1], segmented_coords[indices['elbow'],:,i-1], segmented_coords[indices['shoulder'],:,i], segmented_coords[indices['elbow'],:,i], dt) for i in range(1, num_segmented_frames)]
            elbow_vel = calculate_elbow_extension_velocity(segmented_coords[indices['shoulder']], segmented_coords[indices['elbow']], segmented_coords[indices['wrist']], dt)

        # --- Plotting ---
        st.subheader("運動連鎖グラフ")
        fig, ax = plt.subplots(figsize=(12, 7))
        normalized_time = np.linspace(0, 100, 101)
        ax.plot(normalized_time, normalize_curve(np.abs(pelvis_vel)), label='骨盤 角速度', color='blue')
        ax.plot(normalized_time, normalize_curve(np.abs(thorax_vel)), label='胸郭 角速度', color='green')
        ax.plot(normalized_time, normalize_curve(np.abs(shoulder_vel)), label='肩(上腕) 角速度', color='red')
        ax.plot(normalized_time, normalize_curve(np.abs(elbow_vel)), label='肘(前腕) 伸展速度', color='purple')
        
        ax.set_title('スマートフォン映像からの自動解析結果')
        ax.set_xlabel('正規化時間 (%) [ステップ脚最大挙上～ボールリリース]')
        ax.set_ylabel('角速度の大きさ (deg/s)')
        ax.legend(); ax.grid(True)
        st.pyplot(fig)

        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', dpi=200)
        base_name = os.path.splitext(uploaded_file.name)[0]
        st.download_button(label="グラフをダウンロード", data=img_buf, file_name=f"{base_name}_kinetic_chain.png", mime="image/png")
    else:
        st.error("動画から十分な骨格データを検出できませんでした。撮影条件（明るさ、カメラからの距離、背景など）をご確認の上、別の動画でお試しください。")
    
    # Clean up the temporary video file
    os.remove(uploaded_file.name)