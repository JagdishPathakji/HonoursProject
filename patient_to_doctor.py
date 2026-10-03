"""
Medical interpreter, patient -> doctor.

The patient signs medical words to the webcam, one at a time, lowering the hands to rest between signs
(like the dictionary videos the model learned from). Each recognised word is shown and spoken.
Press Enter and Groq turns the signed words into a proper sentence for the doctor, which is shown and spoken.

Keys: Enter = make sentence for the doctor, Backspace = remove last word, c = clear, q = quit
"""
import os
import sys
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "med_dataset"))
from extract_landmarks import active_frames, frame_features  # noqa: E402
from recognizer import MedSignRecognizer  # noqa: E402
from isl_to_speech import Speaker  # noqa: E402

MIN_CONFIDENCE = 0.5  # below this the sign is shown as "not sure" and not added
REST_SECONDS = 0.5    # hands down this long = sign finished
MIN_SIGN_SECONDS = 0.4
MAX_SIGN_SECONDS = 5.0
PRE_ROLL = 4          # frames kept from before the hands went up


class SignSegmenter:
    """Cuts a live landmark stream into single signs: a sign starts when the hands rise and ends when they rest."""

    def __init__(self, fps):
        self.rest_limit = int(REST_SECONDS * fps)
        self.min_len = int(MIN_SIGN_SECONDS * fps)
        self.max_len = int(MAX_SIGN_SECONDS * fps)
        self.pre = deque(maxlen=PRE_ROLL)
        self.sign = None  # frames of the sign being made
        self.rest = 0     # frames since the hands went down

    @property
    def signing(self):
        return self.sign is not None

    def feed(self, results):
        """Add one MediaPipe Holistic result. Returns the (T, 201) frames of a finished sign, else None."""
        feats, _ = frame_features(results)
        active = bool(results.pose_landmarks) and bool(active_frames(feats[None])[0])

        if self.sign is None:
            self.pre.append(feats)
            if active:
                self.sign, self.rest = list(self.pre), 0
            return None

        self.sign.append(feats)
        self.rest = 0 if active else self.rest + 1
        if self.rest < self.rest_limit and len(self.sign) < self.max_len:
            return None
        return self.flush()

    def flush(self):
        """Finish the sign in progress (e.g. the video ended before the hands came down)."""
        if self.sign is None:
            return None
        frames = np.array(self.sign[:len(self.sign) - self.rest + 2])
        self.sign = None
        self.pre.clear()
        return frames if len(frames) >= self.min_len else None


def words_to_sentence(words):
    """Turn signed ISL words (ISL has its own word order and no grammar words) into a sentence for the doctor."""
    from groq import Groq
    prompt = (
        "A Deaf patient signed these Indian Sign Language words to a doctor, in this order: "
        f"{', '.join(words)}.\n"
        "Write the one short, first-person English sentence the patient most likely means. "
        "Do not add symptoms, times or details that are not in the words. Reply with the sentence only."
    )
    try:
        r = Groq().chat.completions.create(model="openai/gpt-oss-120b", max_tokens=300,
                                           messages=[{"role": "user", "content": prompt}])
        return (r.choices[0].message.content or "").strip() or " ".join(words)
    except Exception as e:  # no internet / API problem: fall back to the raw words
        print(f"Groq unavailable ({e}); using the signed words as they are")
        return " ".join(words)


def draw(frame, words, status, last, sentence):
    H, W, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (W, 40), (0, 0, 0), -1)
    cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if status.startswith("SIGNING") else (0, 255, 0), 2, cv2.LINE_AA)
    if last:
        cv2.putText(frame, last, (W - 330, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, H - 80), (W, H), (0, 0, 0), -1)
    cv2.putText(frame, "Signed: " + " ".join(words)[-45:], (10, H - 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Doctor: " + sentence[-60:], (10, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    load_dotenv(ROOT / ".env")
    recognizer = MedSignRecognizer()
    speaker = Speaker()
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
    print(__doc__)
    print("Words I know:", ", ".join(recognizer.words))

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = min(fps, 30)

    segmenter = SignSegmenter(fps)
    words, last, sentence = [], "", ""

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from webcam")
            break

        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames = segmenter.feed(results)
        if frames is not None:
            top = recognizer.predict(frames)
            word, p = top[0]
            print("Recognised: " + "  ".join(f"{w} {q:.0%}" for w, q in top))
            if p >= MIN_CONFIDENCE:
                words.append(word)
                speaker.say(word)
                last = f"{word} ({p:.0%})"
            else:
                last = f"not sure: {word}? ({p:.0%})"

        # draw the skeleton so the patient can see they are being tracked
        for lms, conn in ((results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS),
                          (results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS),
                          (results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)):
            if lms:
                mp.solutions.drawing_utils.draw_landmarks(frame, lms, conn)
        # mirror only the display (easier to sign to); recognition uses the unmirrored frame like the training videos
        frame = cv2.flip(frame, 1)

        draw(frame, words, "SIGNING..." if segmenter.signing else "Ready - sign a word, then lower hands", last, sentence)
        cv2.imshow("Patient -> Doctor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 13 and words:  # Enter
            sentence = words_to_sentence(words)
            print(f"Doctor: {sentence}")
            speaker.say(sentence)
            words = []
        elif key == 8 and words:  # Backspace
            words.pop()
        elif key == ord('c'):
            words, sentence, last = [], "", ""

    speaker.close()
    holistic.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
