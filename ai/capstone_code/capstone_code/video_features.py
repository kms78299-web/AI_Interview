"""
Video feature 추출.

1순위: OpenFace(FeatureExtraction 바이너리)가 PATH에 있으면 그걸 사용.
2순위: 없으면 MediaPipe Face Mesh(iris 포함)로 대체해서
       AU06/AU12/AU45/AU04는 랜드마크 기반 proxy 값으로, gaze는 iris 위치로,
       pose는 solvePnP로 근사한다.

두 경로 모두 최종적으로 config.VIDEO_FEATURE_NAMES 순서의 1D np.float32
배열(15차원)을 반환한다.
"""

import csv
import os
import shutil
import subprocess
import tempfile

import numpy as np

from config import OPENFACE_BIN, VIDEO_FEATURE_NAMES, VIDEO_FRAME_SAMPLE_STRIDE


def openface_available():
    return shutil.which(OPENFACE_BIN) is not None


def _stack_stats(prefix, values):
    arr = np.asarray(values, dtype=np.float64)
    return {f"{prefix}_mean": float(np.nanmean(arr)), f"{prefix}_std": float(np.nanstd(arr))}


# ─────────────────────────────────────────────
# 1순위: OpenFace
# ─────────────────────────────────────────────
def extract_video_features_openface(video_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            OPENFACE_BIN,
            "-f", video_path,
            "-out_dir", tmp_dir,
            "-aus", "-gaze", "-pose",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"OpenFace 실패 (returncode={result.returncode}): {result.stderr.decode(errors='ignore')[:500]}")

        base = os.path.splitext(os.path.basename(video_path))[0]
        csv_path = os.path.join(tmp_dir, f"{base}.csv")
        if not os.path.exists(csv_path):
            raise RuntimeError(f"OpenFace 출력 CSV를 찾을 수 없음: {csv_path}")

        au06, au12, au45, au04 = [], [], [], []
        gaze_x, gaze_y = [], []
        pose_rx, pose_ry, pose_rz = [], [], []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                if row.get("success", "1") == "0":
                    continue
                au06.append(float(row["AU06_r"]))
                au12.append(float(row["AU12_r"]))
                au45.append(float(row["AU45_r"]))
                au04.append(float(row["AU04_r"]))
                gaze_x.append(float(row["gaze_angle_x"]))
                gaze_y.append(float(row["gaze_angle_y"]))
                pose_rx.append(float(row["pose_Rx"]))
                pose_ry.append(float(row["pose_Ry"]))
                pose_rz.append(float(row["pose_Rz"]))

        if not au06:
            raise RuntimeError("OpenFace가 얼굴을 하나도 인식하지 못함 (success=0)")

        feats = {}
        feats.update(_stack_stats("AU06", au06))
        feats.update(_stack_stats("AU12", au12))
        feats.update(_stack_stats("AU45", au45))
        feats.update(_stack_stats("AU04", au04))
        feats["gaze_x_mean"] = float(np.nanmean(gaze_x))
        feats["gaze_y_mean"] = float(np.nanmean(gaze_y))
        feats["gaze_std"] = float(np.nanmean([np.nanstd(gaze_x), np.nanstd(gaze_y)]))
        feats["pose_rx_mean"] = float(np.nanmean(pose_rx))
        feats["pose_ry_mean"] = float(np.nanmean(pose_ry))
        feats["pose_rz_mean"] = float(np.nanmean(pose_rz))
        feats["pose_std"] = float(np.nanmean([np.nanstd(pose_rx), np.nanstd(pose_ry), np.nanstd(pose_rz)]))

        return np.array([feats[name] for name in VIDEO_FEATURE_NAMES], dtype=np.float32)


# ─────────────────────────────────────────────
# 2순위: MediaPipe Face Mesh (landmark 기반 proxy)
# ─────────────────────────────────────────────
# 468 랜드마크 기준 인덱스. refine_landmarks=True로 468~477(iris)도 같이 나옴.
_L_EYE_OUTER, _L_EYE_INNER = 33, 133
_L_EYE_TOP, _L_EYE_BOTTOM = 159, 145
_R_EYE_OUTER, _R_EYE_INNER = 263, 362
_R_EYE_TOP, _R_EYE_BOTTOM = 386, 374
_L_IRIS_CENTER, _R_IRIS_CENTER = 468, 473
_L_BROW_INNER, _R_BROW_INNER = 55, 285
_L_CHEEK, _R_CHEEK = 50, 280
_MOUTH_LEFT, _MOUTH_RIGHT = 61, 291
_NOSE_TIP, _CHIN = 1, 152

# solvePnP용 일반화된(제너릭) 3D 얼굴 모델 좌표(mm 단위 근사치, 임의 원점).
# 순서: nose_tip, chin, left_eye_outer, right_eye_outer, mouth_left, mouth_right
_MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)


def _eye_aspect_ratio(pts, top, bottom, outer, inner):
    vertical = np.linalg.norm(pts[top] - pts[bottom])
    horizontal = np.linalg.norm(pts[outer] - pts[inner])
    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def _rotation_matrix_to_euler_deg(rmat):
    sy = (rmat[0, 0] ** 2 + rmat[1, 0] ** 2) ** 0.5
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = 0.0
    return np.degrees([pitch, yaw, roll])


def _frame_landmarks_to_features(pts_px, frame_w, frame_h):
    """한 프레임의 픽셀 좌표 랜드마크(478,2) -> AU proxy/gaze/pose dict."""
    import cv2

    inter_ocular = np.linalg.norm(pts_px[_L_EYE_OUTER] - pts_px[_R_EYE_OUTER])
    if inter_ocular < 1e-6:
        return None

    # AU45 (blink) proxy: eye aspect ratio가 작을수록(눈이 감길수록) 값이 커지게 뒤집음.
    ear_l = _eye_aspect_ratio(pts_px, _L_EYE_TOP, _L_EYE_BOTTOM, _L_EYE_OUTER, _L_EYE_INNER)
    ear_r = _eye_aspect_ratio(pts_px, _R_EYE_TOP, _R_EYE_BOTTOM, _R_EYE_OUTER, _R_EYE_INNER)
    au45 = max(0.0, 1.0 - (ear_l + ear_r) / 2.0 * 4.0)  # 대략 0~1 스케일

    # AU12 (lip corner puller / smile) proxy: 입 너비 / 눈 사이 거리.
    mouth_width = np.linalg.norm(pts_px[_MOUTH_LEFT] - pts_px[_MOUTH_RIGHT])
    au12 = mouth_width / inter_ocular

    # AU06 (cheek raiser) proxy: 볼-아래눈꺼풀 거리가 가까워질수록(볼이 올라올수록) 값이 커짐.
    cheek_eye_l = np.linalg.norm(pts_px[_L_CHEEK] - pts_px[_L_EYE_BOTTOM])
    cheek_eye_r = np.linalg.norm(pts_px[_R_CHEEK] - pts_px[_R_EYE_BOTTOM])
    au06 = inter_ocular / max((cheek_eye_l + cheek_eye_r) / 2.0, 1e-6)

    # AU04 (brow lowerer) proxy: 눈썹-눈 거리가 가까워질수록(찌푸릴수록) 값이 커짐.
    brow_eye_l = np.linalg.norm(pts_px[_L_BROW_INNER] - pts_px[_L_EYE_TOP])
    brow_eye_r = np.linalg.norm(pts_px[_R_BROW_INNER] - pts_px[_R_EYE_TOP])
    au04 = inter_ocular / max((brow_eye_l + brow_eye_r) / 2.0, 1e-6)

    # gaze: iris 중심이 눈 중심에서 얼마나 벗어났는지 (양쪽 눈 평균, [-1, 1] 근사)
    l_eye_center = (pts_px[_L_EYE_OUTER] + pts_px[_L_EYE_INNER]) / 2.0
    r_eye_center = (pts_px[_R_EYE_OUTER] + pts_px[_R_EYE_INNER]) / 2.0
    l_eye_w = max(np.linalg.norm(pts_px[_L_EYE_OUTER] - pts_px[_L_EYE_INNER]), 1e-6)
    r_eye_w = max(np.linalg.norm(pts_px[_R_EYE_OUTER] - pts_px[_R_EYE_INNER]), 1e-6)
    l_gaze = (pts_px[_L_IRIS_CENTER] - l_eye_center) / (l_eye_w / 2.0)
    r_gaze = (pts_px[_R_IRIS_CENTER] - r_eye_center) / (r_eye_w / 2.0)
    gaze_x = float((l_gaze[0] + r_gaze[0]) / 2.0)
    gaze_y = float((l_gaze[1] + r_gaze[1]) / 2.0)

    # pose: solvePnP
    image_points = np.array([
        pts_px[_NOSE_TIP], pts_px[_CHIN], pts_px[_L_EYE_OUTER],
        pts_px[_R_EYE_OUTER], pts_px[_MOUTH_LEFT], pts_px[_MOUTH_RIGHT],
    ], dtype=np.float64)
    focal_length = frame_w
    camera_matrix = np.array([
        [focal_length, 0, frame_w / 2.0],
        [0, focal_length, frame_h / 2.0],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    ok, rvec, _tvec = cv2.solvePnP(
        _MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        pose_rx = pose_ry = pose_rz = 0.0
    else:
        rmat, _ = cv2.Rodrigues(rvec)
        pose_rx, pose_ry, pose_rz = _rotation_matrix_to_euler_deg(rmat)

    return {
        "AU06": au06, "AU12": au12, "AU45": au45, "AU04": au04,
        "gaze_x": gaze_x, "gaze_y": gaze_y,
        "pose_rx": float(pose_rx), "pose_ry": float(pose_ry), "pose_rz": float(pose_rz),
    }


def extract_video_features_mediapipe(video_path, frame_stride=VIDEO_FRAME_SAMPLE_STRIDE):
    import cv2
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh

    per_frame = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없음: {video_path}")

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % frame_stride == 0:
                    h, w = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = face_mesh.process(rgb)
                    if result.multi_face_landmarks:
                        landmarks = result.multi_face_landmarks[0].landmark
                        pts_px = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float64)
                        feats = _frame_landmarks_to_features(pts_px, w, h)
                        if feats is not None:
                            per_frame.append(feats)
                frame_idx += 1
    finally:
        cap.release()

    if not per_frame:
        raise RuntimeError(f"얼굴을 하나도 인식하지 못함 (MediaPipe): {video_path}")

    feats = {}
    feats.update(_stack_stats("AU06", [f["AU06"] for f in per_frame]))
    feats.update(_stack_stats("AU12", [f["AU12"] for f in per_frame]))
    feats.update(_stack_stats("AU45", [f["AU45"] for f in per_frame]))
    feats.update(_stack_stats("AU04", [f["AU04"] for f in per_frame]))
    gaze_x_list = [f["gaze_x"] for f in per_frame]
    gaze_y_list = [f["gaze_y"] for f in per_frame]
    feats["gaze_x_mean"] = float(np.nanmean(gaze_x_list))
    feats["gaze_y_mean"] = float(np.nanmean(gaze_y_list))
    feats["gaze_std"] = float(np.nanmean([np.nanstd(gaze_x_list), np.nanstd(gaze_y_list)]))
    pose_rx_list = [f["pose_rx"] for f in per_frame]
    pose_ry_list = [f["pose_ry"] for f in per_frame]
    pose_rz_list = [f["pose_rz"] for f in per_frame]
    feats["pose_rx_mean"] = float(np.nanmean(pose_rx_list))
    feats["pose_ry_mean"] = float(np.nanmean(pose_ry_list))
    feats["pose_rz_mean"] = float(np.nanmean(pose_rz_list))
    feats["pose_std"] = float(np.nanmean([np.nanstd(pose_rx_list), np.nanstd(pose_ry_list), np.nanstd(pose_rz_list)]))

    return np.array([feats[name] for name in VIDEO_FEATURE_NAMES], dtype=np.float32)


def extract_video_features(video_path, frame_stride=VIDEO_FRAME_SAMPLE_STRIDE):
    """OpenFace가 있으면 OpenFace, 없으면 MediaPipe로 자동 분기."""
    if openface_available():
        return extract_video_features_openface(video_path)
    return extract_video_features_mediapipe(video_path, frame_stride=frame_stride)
