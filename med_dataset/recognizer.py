"""Load the trained medical sign model and classify one sign (a sequence of raw landmark frames)."""
from pathlib import Path

import numpy as np
import torch

from features import to_features
from train_model import SignModel

MODEL_PATH = Path(__file__).parent / "model" / "med_sign_model.pt"


class MedSignRecognizer:
    def __init__(self, path=MODEL_PATH):
        ckpt = torch.load(path, weights_only=False)
        self.words = ckpt["words"]
        self.mean, self.std = ckpt["mean"], ckpt["std"]
        self.model = SignModel(ckpt["n_features"], len(self.words))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict(self, frames, top_k=3):
        """frames: (T, 201) raw landmarks of one sign. Returns [(word, probability), ...] best first."""
        x = (to_features(np.asarray(frames)) - self.mean) / self.std
        with torch.no_grad():
            probs = torch.softmax(self.model(torch.from_numpy(x[None]).float()), dim=1)[0].numpy()
        best = np.argsort(probs)[::-1][:top_k]
        return [(self.words[i], float(probs[i])) for i in best]
