"""
Shared MediaPipe + Random Forest gesture pipeline (SRS 1.4.7: reused by Module 3 and Module 4).

    recognizer = GestureRecognizer()
    label = recognizer.predict(frame)   # draws landmarks + label on frame, returns label or None
"""
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MODELS_DIR = Path(__file__).parent / "saved_models"

# Labels for single and double hand signs
DOUBLE_HAND_LABELS = {0: 'A', 1: 'B', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'J', 8: 'K', 9: 'M', 10: 'N',
                      11: 'P', 12: 'Q', 13: 'R', 14: 'S', 15: 'T', 16: 'W', 17: 'X', 18: 'Y', 19: 'Z', 20: 'ACCIDENT', 21: 'HELP'}

SINGLE_HAND_LABELS = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: 'C', 10: 'I',
                      11: 'L', 12: 'O', 13: 'U', 14: 'V', 15: 'PAIN', 16: 'CALL', 17: 'NEXT', 18: 'BACKSPACE', 19: 'SPACE'}


def _load_model(name):
    with open(MODELS_DIR / name, 'rb') as f:
        return pickle.load(f)['model']


class GestureRecognizer:
    def __init__(self):
        self.single_hand_model = _load_model('single_hand_model_word_seq(scikit-upgraded).p')
        self.double_hand_model = _load_model('double_hand_model_word(scikit-upgraded).p')

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

    def predict(self, frame):
        """Detect hands in a BGR frame, draw landmarks and the prediction on it, and return the label (or None)."""
        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None

        num_hands = len(results.multi_hand_landmarks)

        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmark coordinates
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            # Normalize landmarks (same as preprocessing/create_dataset.py, so it matches the training data)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

        # Bounding box
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        try:
            if num_hands == 1:
                prediction = self.single_hand_model.predict([np.asarray(data_aux)])
                label = SINGLE_HAND_LABELS[int(prediction[0])]
            else:
                prediction = self.double_hand_model.predict([np.asarray(data_aux)])
                label = DOUBLE_HAND_LABELS[int(prediction[0])]
        except ValueError as e:
            print(f"Error during prediction: {e}")
            return None

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        return label
