"""
Landmark sequence -> model input. Shared by training (train_model.py) and live recognition,
so both always see exactly the same features.

Raw frame layout (from extract_landmarks.py): 25 pose points, then left hand 21, then right hand 21, each (x, y, z).
"""
import numpy as np

N_POSE, N_HAND = 25, 21
SEQ_LEN = 32  # every sign is resampled to this many frames

# Pose points that matter for signing: nose, eyes, mouth corners, shoulders, elbows, wrists
POSE_KEEP = [0, 2, 5, 9, 10, 11, 12, 13, 14, 15, 16]
L_SHOULDER, R_SHOULDER = 11, 12

# Left/right pairs among pose points 0-24, used when mirroring a sign
POSE_MIRROR = [0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]


def split(frames):
    """(T, 201) -> pose (T, 25, 3), left hand (T, 21, 3), right hand (T, 21, 3)."""
    pts = frames.reshape(len(frames), -1, 3)
    return pts[:, :N_POSE], pts[:, N_POSE:N_POSE + N_HAND], pts[:, N_POSE + N_HAND:]


def join(pose, left, right):
    return np.concatenate([pose, left, right], axis=1).reshape(len(pose), -1)


def resample(frames, length=SEQ_LEN):
    """Linearly resample a (T, F) sequence to `length` frames. Missing hands (zeros) are not interpolated into."""
    T = len(frames)
    if T == 1:
        return np.repeat(frames, length, axis=0)
    pos = np.linspace(0, T - 1, length)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (pos - lo)[:, None]
    out = frames[lo] * (1 - w) + frames[hi] * w
    # if either neighbour had no hand, use the nearest frame instead of blending with zeros
    nearest = frames[np.where(w[:, 0] < 0.5, lo, hi)]
    zero_either = (frames[lo] == 0) | (frames[hi] == 0)
    out[zero_either] = nearest[zero_either]
    return out


def to_features(frames):
    """
    (T, 201) raw landmarks -> (SEQ_LEN, F) features:
      - kept pose points + both hands, relative to the shoulder centre and scaled by shoulder width
        (so position/size of the person in the camera doesn't matter)
      - each hand's shape relative to its own wrist and size (so the hand shape is captured precisely)
      - left/right hand present flags
    Only x, y are used; MediaPipe's z is too noisy for single-camera video.
    """
    frames = resample(np.asarray(frames, dtype=np.float32))
    pose, left, right = split(frames)
    hands_present = np.stack([np.abs(left).sum((1, 2)) > 0, np.abs(right).sum((1, 2)) > 0], axis=1)

    centre = (pose[:, L_SHOULDER, :2] + pose[:, R_SHOULDER, :2]) / 2  # (T, 2)
    width = np.linalg.norm(pose[:, L_SHOULDER, :2] - pose[:, R_SHOULDER, :2], axis=1)
    width = np.maximum(np.median(width), 1e-3)

    def body_rel(p):
        return (p[..., :2] - centre[:, None]) / width

    body = [body_rel(pose[:, POSE_KEEP])]
    shapes = []
    for hand, present in ((left, hands_present[:, 0]), (right, hands_present[:, 1])):
        rel = body_rel(hand)
        rel[~present] = 0
        body.append(rel)
        wrist = hand[:, :1, :2]
        size = np.linalg.norm(hand[:, 9, :2] - hand[:, 0, :2], axis=1)[:, None, None]  # wrist -> middle knuckle
        shape = (hand[..., :2] - wrist) / np.maximum(size, 1e-3)
        shape[~present] = 0
        shapes.append(shape)

    feats = np.concatenate(
        [np.concatenate(body, axis=1).reshape(SEQ_LEN, -1),
         np.concatenate(shapes, axis=1).reshape(SEQ_LEN, -1),
         hands_present.astype(np.float32)], axis=1)
    return feats.astype(np.float32)


N_FEATURES = (len(POSE_KEEP) + 2 * N_HAND) * 2 + 2 * N_HAND * 2 + 2
