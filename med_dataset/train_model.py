"""
Step 3: train the medical sign recogniser (patient -> doctor).

    python med_dataset/train_model.py

There is one reference video per word, so each sign is expanded into many variations that imitate
different people and cameras: speed and timing, start/end point, body size and position, camera tilt,
left-handed signing (mirror), hand-shape wobble, landmark noise and missed hand detections.

Accuracy is reported on:
    val     - fresh variations of the same strength as training (generated with a different seed)
    stress  - variations 1.5x stronger than anything seen in training (a rough "new person" test)
Real-world accuracy on new signers will be lower than both; record real clips to measure and improve it.

Saves med_dataset/model/med_sign_model.pt and med_dataset/model/metrics.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from vocab import WORDS  # noqa: E402
from features import (N_FEATURES, POSE_MIRROR, SEQ_LEN, join,  # noqa: E402
                      split, to_features)

LANDMARKS_DIR = HERE / "landmarks"
MODEL_DIR = HERE / "model"

N_TRAIN, N_VAL, N_STRESS = 400, 60, 60  # variations per word
EPOCHS = 20


# ---------------------------------------------------------------- augmentation

def rotate(points, centre, angle):
    c, s = np.cos(angle), np.sin(angle)
    rel = points - centre
    return centre + rel @ np.array([[c, -s], [s, c]]).T


def augment(frames, segment, rng, strength=1.0):
    """One random variation of a sign. frames: full-video raw landmarks (T, 201); segment: [start, end)."""
    k = strength
    T = len(frames)
    s, e = segment
    L = e - s

    # timing: shift start/end, then non-uniform speed (time warp)
    s2 = int(np.clip(s + rng.uniform(-0.12, 0.12) * k * L, 0, T - 2))
    e2 = int(np.clip(e + rng.uniform(-0.12, 0.12) * k * L, s2 + 2, T))
    seq = frames[s2:e2].copy()
    gamma = np.exp(rng.uniform(-0.3, 0.3) * k)
    idx = np.round((np.linspace(0, 1, len(seq)) ** gamma) * (len(seq) - 1)).astype(int)
    seq = seq[idx]

    pose, left, right = split(seq)
    pose, left, right = pose[..., :2].copy(), left[..., :2].copy(), right[..., :2].copy()
    has_l = np.abs(left).sum((1, 2)) > 0
    has_r = np.abs(right).sum((1, 2)) > 0

    # left-handed signer
    if rng.random() < 0.3:
        pose = pose[:, POSE_MIRROR]
        pose[..., 0] = 1 - pose[..., 0]
        left, right, has_l, has_r = right, left, has_r, has_l
        left[has_l, :, 0] = 1 - left[has_l, :, 0]
        right[has_r, :, 0] = 1 - right[has_r, :, 0]

    # hand-shape wobble: rotate/scale each hand a little around its own wrist
    for hand, has in ((left, has_l), (right, has_r)):
        if has.any():
            wrist = hand[has, :1]
            hand[has] = wrist + (rotate(hand[has], wrist, rng.normal(0, 0.12 * k)) - wrist) * rng.uniform(1 - 0.1 * k, 1 + 0.1 * k)

    # body/camera: tilt, size, aspect, position (applied to everything)
    centre = pose[:, 11:13].mean(axis=(0, 1))
    angle = rng.normal(0, 0.12 * k)
    scale = rng.uniform(1 - 0.15 * k, 1 + 0.15 * k)
    aspect = np.array([rng.uniform(1 - 0.08 * k, 1 + 0.08 * k), 1.0])
    shift = rng.normal(0, 0.05 * k, size=2)

    def camera(p):
        q = (rotate(p, centre, angle) - centre) * scale * aspect + centre + shift
        # landmark noise + slow drift
        q += rng.normal(0, 0.004 * k, size=q.shape)
        return q

    pose = camera(pose)
    for hand, has in ((left, has_l), (right, has_r)):
        if has.any():
            hand[has] = camera(hand[has])

    # missed hand detections: random frames + sometimes a longer gap
    for hand, has in ((left, has_l), (right, has_r)):
        drop = rng.random(len(hand)) < rng.uniform(0, 0.2 * k)
        if rng.random() < 0.3 * k:
            g = rng.integers(0, len(hand))
            drop[g:g + rng.integers(2, max(3, len(hand) // 4))] = True
        hand[drop & has] = 0

    z = lambda p: np.concatenate([p, np.zeros(p.shape[:-1] + (1,), np.float32)], axis=-1)  # noqa: E731
    return join(z(pose), z(left), z(right))


def build_set(signs, n_per_word, seed, strength):
    rng = np.random.default_rng(seed)
    X = np.empty((len(signs) * n_per_word, SEQ_LEN, N_FEATURES), np.float32)
    y = np.empty(len(X), np.int64)
    i = 0
    for label, (frames, segment) in enumerate(signs):
        for _ in range(n_per_word):
            X[i] = to_features(augment(frames, segment, rng, strength))
            y[i] = label
            i += 1
    return X, y


# ---------------------------------------------------------------- model

class SignModel(nn.Module):
    """Temporal conv (local motion) -> bidirectional GRU (whole-sign order) -> mean+max pool -> classes."""

    def __init__(self, n_features, n_classes, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * hidden, n_classes))

    def forward(self, x):  # x: (B, T, F)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(torch.cat([h.mean(1), h.amax(1)], dim=1))


def evaluate(model, X, y, mean, std):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy((X - mean) / std)).argmax(1).numpy()
    return float((pred == y).mean()), pred


def main():
    torch.manual_seed(0)
    signs = []
    for w in WORDS:
        d = np.load(LANDMARKS_DIR / f"{w.replace(' ', '_')}.npz")
        signs.append((d["frames"], tuple(d["segment"])))

    t0 = time.time()
    X_tr, y_tr = build_set(signs, N_TRAIN, seed=1, strength=1.0)
    X_va, y_va = build_set(signs, N_VAL, seed=2, strength=1.0)
    X_st, y_st = build_set(signs, N_STRESS, seed=3, strength=1.5)
    print(f"built {len(X_tr)} train / {len(X_va)} val / {len(X_st)} stress sequences in {time.time() - t0:.0f}s")

    mean = X_tr.reshape(-1, N_FEATURES).mean(0)
    std = X_tr.reshape(-1, N_FEATURES).std(0) + 1e-6

    model = SignModel(N_FEATURES, len(WORDS))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EPOCHS * (len(X_tr) // 128 + 1))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xt = torch.from_numpy((X_tr - mean) / std)
    yt = torch.from_numpy(y_tr)

    best, best_state = 0.0, None
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 128):
            idx = perm[b:b + 128]
            xb = Xt[idx] + 0.05 * torch.randn_like(Xt[idx])
            loss = loss_fn(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        val_acc, _ = evaluate(model, X_va, y_va, mean, std)
        if val_acc >= best:
            best, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 4 or epoch == EPOCHS - 1:
            print(f"epoch {epoch + 1:2d}  loss {loss.item():.3f}  val {val_acc:.1%}")

    model.load_state_dict(best_state)
    val_acc, _ = evaluate(model, X_va, y_va, mean, std)
    stress_acc, stress_pred = evaluate(model, X_st, y_st, mean, std)

    per_word = {w: float((stress_pred[y_st == i] == i).mean()) for i, w in enumerate(WORDS)}
    confusions = {}
    for i, w in enumerate(WORDS):
        wrong = stress_pred[(y_st == i) & (stress_pred != i)]
        if len(wrong):
            top = np.bincount(wrong, minlength=len(WORDS)).argmax()
            confusions[w] = WORDS[top]

    print(f"\nval accuracy    {val_acc:.1%}")
    print(f"stress accuracy {stress_acc:.1%}")
    weakest = sorted(per_word.items(), key=lambda kv: kv[1])[:8]
    print("weakest words (stress):", ", ".join(f"{w} {a:.0%} (-> {confusions.get(w, '-')})" for w, a in weakest))

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "words": WORDS, "mean": mean, "std": std,
                "n_features": N_FEATURES, "seq_len": SEQ_LEN}, MODEL_DIR / "med_sign_model.pt")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(
        {"val_accuracy": val_acc, "stress_accuracy": stress_acc, "per_word_stress": per_word,
         "most_confused_with": confusions}, indent=2))
    print(f"saved {MODEL_DIR / 'med_sign_model.pt'}")


if __name__ == "__main__":
    main()
