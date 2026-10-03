"""
Step 2: extract MediaPipe Holistic landmarks from every downloaded video.

    python med_dataset/extract_landmarks.py            # all words, skips ones already done
    python med_dataset/extract_landmarks.py --force    # redo everything

For each video saves med_dataset/landmarks/<word>.npz with:
    frames   (T, 201) float32  per frame: 25 upper-body pose points + left hand 21 + right hand 21, each (x, y, z).
                               A missing hand is all zeros.
    hands    (T, 2)   bool     left / right hand detected
    active   (T,)     bool     hands raised (signing) vs resting
    segment  (2,)     int      [start, end) of the first signing segment = the sign itself.
                               Long dictionary videos continue with an explanation after it; that part is ignored.
    fps      float
"""
import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from vocab import WORDS  # noqa: E402

VIDEOS_DIR = HERE / "videos"
LANDMARKS_DIR = HERE / "landmarks"

N_POSE = 25  # pose landmarks 0-24: face, shoulders, arms, hands, hips (legs are out of frame)
N_HAND = 21
N_FEATURES = (N_POSE + 2 * N_HAND) * 3

# Pose indices
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 11, 12, 15, 16, 23, 24


def frame_features(results):
    feats = np.zeros(N_FEATURES, dtype=np.float32)
    hands = np.zeros(2, dtype=bool)
    if results.pose_landmarks:
        feats[:N_POSE * 3] = [v for lm in results.pose_landmarks.landmark[:N_POSE] for v in (lm.x, lm.y, lm.z)]
    for i, hand in enumerate([results.left_hand_landmarks, results.right_hand_landmarks]):
        if hand:
            start = (N_POSE + i * N_HAND) * 3
            feats[start:start + N_HAND * 3] = [v for lm in hand.landmark for v in (lm.x, lm.y, lm.z)]
            hands[i] = True
    return feats, hands


def pose_point(frames, idx):
    return frames[:, idx * 3:idx * 3 + 2]  # (T, 2) x, y


def active_frames(frames):
    """A frame is 'signing' when either wrist is raised well above the hips (resting hands hang at hip level)."""
    shoulder_y = (pose_point(frames, L_SHOULDER)[:, 1] + pose_point(frames, R_SHOULDER)[:, 1]) / 2
    hip_y = (pose_point(frames, L_HIP)[:, 1] + pose_point(frames, R_HIP)[:, 1]) / 2
    threshold = hip_y - 0.25 * (hip_y - shoulder_y)
    wrist_y = np.minimum(pose_point(frames, L_WRIST)[:, 1], pose_point(frames, R_WRIST)[:, 1])
    return wrist_y < threshold


def active_runs(active, fps, max_gap_s=0.6, min_len_s=0.4):
    """Runs of active frames as (start, end), bridging gaps shorter than max_gap_s, dropping runs shorter than min_len_s."""
    max_gap, min_len = int(max_gap_s * fps), int(min_len_s * fps)
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return []
    runs, start, end = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - end > max_gap:
            runs.append((start, end + 1))
            start = i
        end = i
    runs.append((start, end + 1))
    return [r for r in runs if r[1] - r[0] >= min_len]


def wrist_speed(frames):
    speed = np.zeros(len(frames))
    for idx in (L_WRIST, R_WRIST):
        p = pose_point(frames, idx)
        speed[1:] += np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.convolve(speed, np.ones(5) / 5, mode="same")  # smooth over 5 frames


def first_segment(frames, active, fps, max_sign_s=4.0):
    """The sign: the first signing run, cut short if the video goes straight on into an explanation."""
    runs = active_runs(active, fps)
    if not runs:
        return 0, len(active)
    # Some clips begin mid-movement (left over from the previous clip). Skip that if a real run follows.
    if runs[0][0] == 0 and len(runs) > 1:
        runs = runs[1:]
    start, end = runs[0]
    # No rest between the sign and the explanation: cut at the calmest moment 1.5-4 s into the run.
    if end - start > max_sign_s * fps:
        lo, hi = start + int(1.5 * fps), start + int(max_sign_s * fps)
        end = lo + int(np.argmin(wrist_speed(frames)[lo:hi]))
    # pad a few frames so the start/end of the movement is kept
    pad = int(0.2 * fps)
    return max(0, start - pad), min(len(active), end + pad)


def extract(video_path, holistic):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, hands = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * 960 / w)))
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        f, hd = frame_features(results)
        frames.append(f)
        hands.append(hd)
    cap.release()
    frames, hands = np.array(frames), np.array(hands)
    active = active_frames(frames)
    segment = np.array(first_segment(frames, active, fps))
    return dict(frames=frames, hands=hands, active=active, segment=segment, fps=fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-extract words that are already done")
    args = parser.parse_args()

    LANDMARKS_DIR.mkdir(exist_ok=True)
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)

    print(f"{'word':15} {'frames':>6} {'sign(s)':>12} {'hands found':>11}")
    for word in WORDS:
        name = word.replace(" ", "_")
        video, out = VIDEOS_DIR / f"{name}.mp4", LANDMARKS_DIR / f"{name}.npz"
        if not video.exists():
            print(f"{word:15} missing video, run download_videos.py first")
            continue
        if out.exists() and not args.force:
            continue
        data = extract(video, holistic)
        np.savez_compressed(out, **data)
        s, e = data["segment"]
        hand_rate = data["hands"][s:e].any(axis=1).mean()
        print(f"{word:15} {len(data['frames']):6d} {s / data['fps']:5.1f}-{e / data['fps']:4.1f}s {hand_rate:10.0%}")

    holistic.close()


if __name__ == "__main__":
    main()
