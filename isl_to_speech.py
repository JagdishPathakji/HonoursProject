"""
Module 4: Live ISL -> Speech.

Uses the same MediaPipe + Random Forest pipeline as Module 3 (gesture_recognizer.py) and speaks
the result with pyttsx3 (offline Windows/Linux TTS).

A gesture is accepted once it is held steady for HOLD_FRAMES frames. To repeat the same sign,
drop your hand (or change sign) and sign it again.

Gestures:
    letters / digits      -> added to the current word
    SPACE                 -> speak the current word
    BACKSPACE             -> delete the last letter
    PAIN, CALL, HELP, ... -> spoken immediately as a word
    NEXT                  -> speak the whole sentence, then start a new one

Keys: space = speak word, enter = speak sentence, backspace = delete letter, c = clear, q = quit
"""
import queue
import threading

import cv2
import pyttsx3

from gesture_recognizer import GestureRecognizer

HOLD_FRAMES = 12  # consecutive identical predictions needed to accept a gesture


class Speaker:
    """Speaks text on a background thread so the webcam loop never freezes.

    pyttsx3 only works reliably when one engine is created and reused on the same thread.
    """

    def __init__(self, rate=160):
        self.queue = queue.Queue()
        self.rate = rate
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', self.rate)
        while True:
            text = self.queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()

    def say(self, text):
        text = text.strip()
        if text:
            print(f"Speaking: {text}")
            self.queue.put(text)

    def close(self):
        self.queue.put(None)


class SentenceBuilder:
    """Turns a stream of accepted gestures into words and a sentence, speaking as it goes."""

    def __init__(self, speaker):
        self.speaker = speaker
        self.current_word = ""
        self.sentence = []

    def finish_word(self):
        if self.current_word:
            self.speaker.say(self.current_word)
            self.sentence.append(self.current_word)
            self.current_word = ""

    def speak_sentence(self):
        self.finish_word()
        self.speaker.say(" ".join(self.sentence))
        self.sentence = []

    def backspace(self):
        self.current_word = self.current_word[:-1]

    def clear(self):
        self.current_word = ""
        self.sentence = []

    def add(self, label):
        if label == 'SPACE':
            self.finish_word()
        elif label == 'BACKSPACE':
            self.backspace()
        elif label == 'NEXT':
            self.speak_sentence()
        elif len(label) == 1:
            self.current_word += label
        else:
            # Whole-word sign (PAIN, CALL, HELP, ACCIDENT)
            self.finish_word()
            self.speaker.say(label.lower())
            self.sentence.append(label.lower())


class GestureStabilizer:
    """Accepts a label only after it is predicted HOLD_FRAMES times in a row, and only once per hold."""

    def __init__(self, hold_frames=HOLD_FRAMES):
        self.hold_frames = hold_frames
        self.last = None
        self.count = 0
        self.accepted = False

    def update(self, label):
        """Returns the label when it becomes accepted, else None. Also returns hold progress 0..1."""
        if label != self.last:
            self.last = label
            self.count = 0
            self.accepted = False
        if label is None:
            return None, 0.0

        self.count += 1
        if self.count >= self.hold_frames and not self.accepted:
            self.accepted = True
            return label, 1.0
        return None, min(self.count / self.hold_frames, 1.0)


def draw_overlay(frame, builder, progress, label):
    H, W, _ = frame.shape
    cv2.rectangle(frame, (0, H - 90), (W, H), (0, 0, 0), -1)
    cv2.putText(frame, "Sentence: " + " ".join(builder.sentence)[-40:], (10, H - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Word: " + builder.current_word, (10, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    # Hold-progress bar for the gesture currently being held
    if label:
        cv2.rectangle(frame, (W - 160, H - 35), (W - 160 + int(150 * progress), H - 20), (0, 200, 0), -1)
        cv2.rectangle(frame, (W - 160, H - 35), (W - 10, H - 20), (255, 255, 255), 1)


def main():
    recognizer = GestureRecognizer()
    speaker = Speaker()
    builder = SentenceBuilder(speaker)
    stabilizer = GestureStabilizer()

    cap = cv2.VideoCapture(0)
    print(__doc__)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from webcam")
            break

        label = recognizer.predict(frame)
        accepted, progress = stabilizer.update(label)
        if accepted:
            print(f"Accepted: {accepted}")
            builder.add(accepted)

        draw_overlay(frame, builder, progress, label)
        cv2.imshow('ISL to Speech', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            builder.finish_word()
        elif key == 13:  # Enter
            builder.speak_sentence()
        elif key == 8:  # Backspace
            builder.backspace()
        elif key == ord('c'):
            builder.clear()

    speaker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
