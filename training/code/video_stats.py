"""
Video 통계 추출 (Track 2: LLM 피드백용, 사람이 읽을 수 있는 값).

MediaPipe FaceMesh(눈동자 포함, 478점)로 영상을 초당 몇 프레임씩 훑어서 얼굴 움직임/시선/고개 방향/눈 깜빡임/입 모양을 요약한다.
MLP 입력(예측용 임베딩)과는 별개이고, 예측 정확도를 올리려는 것이 아니라 LLM이 "관찰된 패턴"을 설명하게 하려는 값이다
(MediaPipe 얼굴 통계는 점수 예측에는 도움이 안 됐음: test Spearman -0.005).

모든 값은 얼굴 크기에 맞춰 정규화한 상대값이라 카메라 거리에 덜 민감하다. 각도(도)가 아니라 근사 지표이므로
'절대 각도'가 아니라 '얼마나 많이/자주'를 해석하는 용도로만 쓴다.

반환 (모두 float):
  face_detected_ratio     얼굴이 잡힌 프레임 비율 (낮으면 아래 값들을 믿기 어려움)
  gaze_off_center_mean    시선이 정면에서 벗어난 정도 평균 (0=정면, 클수록 옆/위/아래를 봄)
  gaze_away_ratio         시선이 크게 벗어난 프레임 비율
  gaze_x_std, gaze_y_std  시선이 좌우/상하로 흔들린 정도
  head_yaw_mean_abs       고개가 좌우로 돌아간 정도 평균 (0=정면)
  head_pitch_mean         고개 상하 기울기 평균 (+ 아래, - 위; 개인 기준선 대비 해석)
  head_roll_std           고개가 갸웃거린 정도 변동
  head_motion             프레임 간 머리 위치 변화 평균 (클수록 많이 움직임)
  blink_rate_per_min      분당 눈 깜빡임 수 (근사)
  mouth_open_std          입을 벌렸다 닫는 변동 (말할 때 입 움직임)
  mouth_width_std         입 가로폭 변동 (표정 변화의 근사)
  n_frames_used           분석에 쓴 프레임 수
"""

import math

import numpy as np

# FaceMesh 랜드마크 번호 (refine_landmarks=True: 468~477이 눈동자)
R_EYE_OUTER, R_EYE_INNER, R_UP, R_LOW, R_IRIS = 33, 133, 159, 145, 468   # 피사체의 오른쪽 눈
L_EYE_INNER, L_EYE_OUTER, L_UP, L_LOW, L_IRIS = 362, 263, 386, 374, 473  # 피사체의 왼쪽 눈
NOSE, MOUTH_L, MOUTH_R, LIP_UP, LIP_LOW = 1, 61, 291, 13, 14

GAZE_AWAY_THRESHOLD = 0.2  # 눈동자가 눈 폭 중앙에서 20% 이상 벗어나면 "크게 벗어남"으로 센다


def _frame_features(lm, w, h):
    """랜드마크 한 프레임 -> 지표 dict. lm: (478, 3) 정규화 좌표."""
    p = lambda i: np.array([lm[i, 0] * w, lm[i, 1] * h])  # noqa: E731
    r_out, r_in, l_in, l_out = p(R_EYE_OUTER), p(R_EYE_INNER), p(L_EYE_INNER), p(L_EYE_OUTER)
    eye_dist = np.linalg.norm(r_out - l_out)
    if eye_dist < 1e-6:
        return None

    def ratio(iris, a, b):  # 눈동자가 두 눈꼬리 사이의 어디에 있나 (0~1, 0.5=중앙)
        v = b - a
        return float(np.dot(p(iris) - a, v) / (np.dot(v, v) + 1e-9))

    gx = (ratio(R_IRIS, r_out, r_in) + ratio(L_IRIS, l_in, l_out)) / 2 - 0.5

    def vratio(iris, up, low):
        return float((p(iris)[1] - p(up)[1]) / (p(low)[1] - p(up)[1] + 1e-9))

    gy = (vratio(R_IRIS, R_UP, R_LOW) + vratio(L_IRIS, L_UP, L_LOW)) / 2 - 0.5

    mid_eyes = (r_out + l_out) / 2
    nose = p(NOSE)
    eye_open = (np.linalg.norm(p(R_UP) - p(R_LOW)) / (np.linalg.norm(r_out - r_in) + 1e-9)
                + np.linalg.norm(p(L_UP) - p(L_LOW)) / (np.linalg.norm(l_out - l_in) + 1e-9)) / 2
    return {
        "gx": gx, "gy": gy,
        "yaw": float((nose[0] - mid_eyes[0]) / eye_dist),      # 코가 두 눈 중앙에서 좌우로 얼마나 치우쳤나
        "pitch": float((nose[1] - mid_eyes[1]) / eye_dist),    # 상하
        "roll": float(math.atan2(l_out[1] - r_out[1], l_out[0] - r_out[0])),  # 두 눈을 잇는 선의 기울기(라디안)
        "nose_xy": nose / eye_dist,
        "eye_open": float(eye_open),
        "mouth_open": float(np.linalg.norm(p(LIP_UP) - p(LIP_LOW)) / eye_dist),
        "mouth_width": float(np.linalg.norm(p(MOUTH_L) - p(MOUTH_R)) / eye_dist),
    }


def extract_video_stats(video_path, sample_fps=5.0, max_seconds=180.0):
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없음: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / sample_fps)))
    real_sample_fps = fps / step
    limit = int(max_seconds * fps)

    feats, n_seen = [], 0
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    try:
        idx = 0
        while idx < limit and (n_total == 0 or idx < n_total):
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                n_seen += 1
                h, w = frame.shape[:2]
                res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if res.multi_face_landmarks:
                    lm = np.array([[l.x, l.y, l.z] for l in res.multi_face_landmarks[0].landmark])
                    f = _frame_features(lm, w, h)
                    if f is not None:
                        feats.append(f)
            idx += 1
    finally:
        mesh.close()
        cap.release()

    if len(feats) < 3:
        raise RuntimeError(f"얼굴이 거의 검출되지 않음 (샘플 {n_seen}프레임 중 {len(feats)}개): {video_path}")

    col = lambda k: np.array([f[k] for f in feats])  # noqa: E731
    gx, gy = col("gx"), col("gy")
    off_center = np.sqrt(gx ** 2 + gy ** 2)
    nose_xy = np.stack([f["nose_xy"] for f in feats])
    motion = np.linalg.norm(np.diff(nose_xy, axis=0), axis=1) if len(feats) > 1 else np.zeros(1)

    eye_open = col("eye_open")
    closed = eye_open < 0.6 * np.median(eye_open)  # 그 사람의 평소 눈 뜬 정도보다 크게 줄면 깜빡임으로 간주
    blinks = int(np.sum(closed[1:] & ~closed[:-1])) + int(closed[0])
    duration_min = len(feats) / real_sample_fps / 60.0

    return {
        "face_detected_ratio": float(len(feats) / max(n_seen, 1)),
        "gaze_off_center_mean": float(off_center.mean()),
        "gaze_away_ratio": float((off_center > GAZE_AWAY_THRESHOLD).mean()),
        "gaze_x_std": float(gx.std()), "gaze_y_std": float(gy.std()),
        "head_yaw_mean_abs": float(np.abs(col("yaw")).mean()),
        "head_pitch_mean": float(col("pitch").mean()),
        "head_roll_std": float(col("roll").std()),
        "head_motion": float(motion.mean()),
        "blink_rate_per_min": float(blinks / duration_min) if duration_min > 0 else 0.0,
        "mouth_open_std": float(col("mouth_open").std()),
        "mouth_width_std": float(col("mouth_width").std()),
        "n_frames_used": float(len(feats)),
    }
