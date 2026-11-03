"""
Video 통계 추출 (LLM 피드백용). MediaPipe FaceMesh로 영상을 초당 5프레임 읽어 시선, 고개 방향, 눈 깜빡임, 입 모양을 요약한다.

값은 얼굴 크기로 정규화한 근사 지표라서 각도가 아니라 얼마나 많이/자주를 읽는 용도다. MLP 입력이 아니다.
반환 키: face_detected_ratio, gaze_off_center_mean, gaze_away_ratio, gaze_x_std, gaze_y_std, head_yaw_mean_abs,
head_pitch_mean, head_roll_std, head_motion, blink_rate_per_min, mouth_open_std, mouth_width_std, n_frames_used
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


FALLBACK_FPS = 30.0  # 타임스탬프를 못 읽을 때(0이거나 늘어나지 않을 때)만 쓰는 가정값


def _frame_times(cap):
    """cap에서 프레임을 하나씩 넘기며 (번호, 실제 시간(초))을 낸다. 브라우저 WebM은 FPS와 프레임 수 메타데이터가 틀려서 CAP_PROP_POS_MSEC만 쓴다."""
    import cv2
    prev_t, idx = None, 0
    while cap.grab():
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if prev_t is None:
            t = max(t, 0.0)
        elif not t > prev_t:
            t = prev_t + 1.0 / FALLBACK_FPS
        yield idx, t
        prev_t, idx = t, idx + 1


def video_duration_sec(video_path):
    """영상 길이(초): 마지막 프레임의 실제 타임스탬프 기준. 메타데이터(FPS, 프레임 수)는 쓰지 않는다. 읽을 수 없으면 0.0."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    try:
        last = 0.0
        for _, t in _frame_times(cap):
            last = t
        return float(last)
    finally:
        cap.release()


def iter_sampled_frames(video_path, sample_fps=5.0, max_seconds=180.0):
    """영상을 실제 시간 기준으로 sample_fps 간격마다 한 프레임씩 낸다: (시간(초), BGR 프레임). max_seconds를 넘으면 멈춘다."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없음: {video_path}")
    interval, next_t = 1.0 / sample_fps, 0.0
    try:
        for _, t in _frame_times(cap):
            if t > max_seconds:
                break
            if t >= next_t:
                ok, frame = cap.retrieve()
                if ok:
                    yield t, frame
                next_t = t + interval
    finally:
        cap.release()


def extract_video_stats(video_path, sample_fps=5.0, max_seconds=180.0):
    import cv2
    import mediapipe as mp

    feats, times, n_seen = [], [], 0
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    try:
        for t, frame in iter_sampled_frames(video_path, sample_fps, max_seconds):
            n_seen += 1
            h, w = frame.shape[:2]
            res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = np.array([[l.x, l.y, l.z] for l in res.multi_face_landmarks[0].landmark])
                f = _frame_features(lm, w, h)
                if f is not None:
                    feats.append(f)
                    times.append(t)
    finally:
        mesh.close()

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
    span = times[-1] - times[0]  # 얼굴이 잡힌 첫 샘플부터 마지막 샘플까지의 실제 시간(초)
    duration_min = (span if span > 0 else len(feats) / sample_fps) / 60.0

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
