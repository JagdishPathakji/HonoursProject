"""
ISL medical interpreter: one screen for a Deaf patient and a hearing doctor.

    python interpreter.py
    python interpreter.py --video clip.mp4   # use a recorded video as the patient's camera (testing / demos)

Left:  the patient's webcam. The patient signs medical words one at a time, lowering the hands between signs.
       Each recognised word is shown; Enter turns them into a sentence for the doctor (shown + spoken).
Right: the doctor's reply in sign language. The doctor presses Tab and types the reply, then Enter,
       or presses m to speak it (6 s, English/Hindi/other Indian languages).
Bottom: the conversation so far. It is saved to conversations/ when you quit.

Keys (patient mode): Enter = send signed words to doctor, Backspace = remove last word, c = clear words,
                     Tab = doctor types, m = doctor speaks, r = replay last reply, q = quit
Keys (typing mode):  type the reply, Enter = send, Esc = cancel
"""
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "med_dataset"))
from isl_to_speech import Speaker  # noqa: E402
from patient_to_doctor import MIN_CONFIDENCE, SignSegmenter, words_to_sentence  # noqa: E402
from recognizer import MedSignRecognizer  # noqa: E402
from sign_player import text_to_signs, video_path  # noqa: E402

PANE_W, PANE_H = 640, 480
LOG_H = 170
FONT = cv2.FONT_HERSHEY_SIMPLEX


class SignPane:
    """Plays a reply's sign videos inside the right pane, one frame per call, without blocking the webcam."""

    def __init__(self):
        self.clips = []  # [(token, VideoCapture)]
        self.index = 0
        self.caption = ""
        self.next_frame_at = 0.0
        self.frame = None
        self.frame_delay = 1 / 25
        self.last = None  # (tokens, paths, caption) for replay

    def start(self, tokens, paths, caption):
        self.last = (tokens, paths, caption)
        self.clips = [(t, cv2.VideoCapture(str(paths[t]))) for t in tokens if paths.get(t)]
        self.index, self.caption, self.frame = 0, caption, None

    def replay(self):
        if self.last:
            self.start(*self.last)

    @property
    def playing(self):
        return self.index < len(self.clips)

    def render(self):
        pane = np.full((PANE_H, PANE_W, 3), 40, np.uint8)
        if self.playing and time.time() >= self.next_frame_at:
            token, cap = self.clips[self.index]
            ok, frame = cap.read()
            if ok:
                self.frame = cv2.resize(frame, (PANE_W, PANE_H - 70))
                self.frame_delay = 1 / (cap.get(cv2.CAP_PROP_FPS) or 25)
            else:
                cap.release()
                self.index += 1
            self.next_frame_at = time.time() + self.frame_delay
        if self.frame is not None:
            pane[:PANE_H - 70] = self.frame
        if self.playing:
            token = self.clips[self.index][0]
            cv2.putText(pane, token.upper(), (10, PANE_H - 42), FONT, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(pane, f"{self.index + 1}/{len(self.clips)}", (PANE_W - 70, PANE_H - 42), FONT, 0.6, (200, 200, 200), 1)
        elif self.frame is None:
            cv2.putText(pane, "Doctor's reply will be signed here", (110, PANE_H // 2), FONT, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        if self.caption:
            cv2.putText(pane, self.caption[:75], (10, PANE_H - 14), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return pane


class Interpreter:
    def __init__(self):
        self.recognizer = MedSignRecognizer()
        self.speaker = Speaker()
        self.holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
        self.pane = SignPane()
        self.events = queue.Queue()  # results from background threads: (kind, payload)
        self.words, self.last_sign, self.status = [], "", ""
        self.typing, self.typed = False, ""
        self.busy = False
        self.log = []  # [(time, who, text)]

    # ------------------------------------------------------------------ background work

    def _background(self, fn, *args):
        self.busy = True

        def run():
            try:
                fn(*args)
            except Exception as e:
                self.events.put(("status", f"Error: {e}"))
            finally:
                self.busy = False
        threading.Thread(target=run, daemon=True).start()

    def _patient_sentence(self, words):
        self.events.put(("status", "Making sentence for the doctor..."))
        self.events.put(("patient", words_to_sentence(words)))

    def _doctor_reply(self, text):
        self.events.put(("status", "Translating reply to ISL..."))
        tokens = text_to_signs(text)
        self.events.put(("status", f"Loading {len(set(tokens))} sign videos..."))
        paths = {t: video_path(t) for t in dict.fromkeys(tokens)}
        self.events.put(("doctor", (text, tokens, paths)))

    def _doctor_speaks(self):
        from doctor_to_patient import listen
        self.events.put(("status", "Doctor, speak now (6 seconds)..."))
        text = listen()
        if text:
            self._doctor_reply(text)
        else:
            self.events.put(("status", "Didn't catch that - press m to try again"))

    def handle_events(self):
        while not self.events.empty():
            kind, payload = self.events.get()
            if kind == "status":
                self.status = payload
            elif kind == "patient":
                self.log.append((datetime.now(), "Patient", payload))
                self.speaker.say(payload)
                self.status = ""
            elif kind == "doctor":
                text, tokens, paths = payload
                self.log.append((datetime.now(), "Doctor", f"{text}   [signed: {' '.join(tokens)}]"))
                self.pane.start(tokens, paths, text)
                self.status = ""

    # ------------------------------------------------------------------ keys

    def on_key(self, key):
        """Returns False to quit."""
        if key == 255:
            return True
        if self.typing:
            if key == 27:  # Esc
                self.typing, self.typed = False, ""
            elif key == 13:  # Enter
                if self.typed.strip():
                    self._background(self._doctor_reply, self.typed.strip())
                self.typing, self.typed = False, ""
            elif key == 8:
                self.typed = self.typed[:-1]
            elif 32 <= key < 127:
                self.typed += chr(key)
            return True

        if key == ord("q"):
            return False
        if key == 9:  # Tab
            self.typing = True
        elif key == ord("m") and not self.busy:
            self._background(self._doctor_speaks)
        elif key == ord("r"):
            self.pane.replay()
        elif key == 13 and self.words and not self.busy:
            self._background(self._patient_sentence, self.words)
            self.words = []
        elif key == 8 and self.words:
            self.words.pop()
        elif key == ord("c"):
            self.words, self.last_sign = [], ""
        return True

    # ------------------------------------------------------------------ drawing

    def draw_patient(self, frame, signing):
        frame = cv2.resize(frame, (PANE_W, PANE_H))
        cv2.rectangle(frame, (0, 0), (PANE_W, 36), (0, 0, 0), -1)
        label = "SIGNING..." if signing else "Patient: sign a word, then lower hands"
        cv2.putText(frame, label, (10, 25), FONT, 0.6, (0, 0, 255) if signing else (0, 255, 0), 2, cv2.LINE_AA)
        if self.last_sign:
            cv2.putText(frame, self.last_sign, (PANE_W - 240, 25), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, PANE_H - 36), (PANE_W, PANE_H), (0, 0, 0), -1)
        cv2.putText(frame, "Signed: " + " ".join(self.words)[-48:], (10, PANE_H - 12), FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return frame

    def draw_log(self):
        panel = np.full((LOG_H, 2 * PANE_W, 3), 25, np.uint8)
        y = 22
        if self.typing:
            cv2.putText(panel, "Doctor typing: " + self.typed + "_", (10, y), FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        elif self.status:
            cv2.putText(panel, self.status, (10, y), FONT, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Enter: send signs | Tab: doctor types | m: doctor speaks | r: replay | q: quit",
                        (10, y), FONT, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        for t, who, text in self.log[-5:]:
            y += 28
            colour = (120, 255, 120) if who == "Patient" else (255, 200, 120)
            cv2.putText(panel, f"{who}: {text}"[:110], (10, y), FONT, 0.5, colour, 1, cv2.LINE_AA)
        return panel

    # ------------------------------------------------------------------ main loop

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        fps = min(cap.get(cv2.CAP_PROP_FPS) or 30, 30)
        segmenter = SignSegmenter(fps)
        print(__doc__)
        print("Patient signs I know:", ", ".join(self.recognizer.words))

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read from webcam")
                break

            results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames = segmenter.feed(results)
            if frames is not None:
                word, p = self.recognizer.predict(frames)[0]
                if p >= MIN_CONFIDENCE:
                    self.words.append(word)
                    self.speaker.say(word)
                    self.last_sign = f"{word} ({p:.0%})"
                else:
                    self.last_sign = f"not sure: {word}? ({p:.0%})"

            for lms, conn in ((results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS),
                              (results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS),
                              (results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)):
                if lms:
                    mp.solutions.drawing_utils.draw_landmarks(frame, lms, conn)
            frame = cv2.flip(frame, 1)  # mirror the display only; recognition used the unmirrored frame

            self.handle_events()
            screen = np.vstack([np.hstack([self.draw_patient(frame, segmenter.signing), self.pane.render()]),
                                self.draw_log()])
            cv2.imshow("ISL Medical Interpreter", screen)
            if not self.on_key(cv2.waitKey(1) & 0xFF):
                break

        self.speaker.close()
        self.holistic.close()
        cap.release()
        cv2.destroyAllWindows()
        self.save_log()

    def save_log(self):
        if not self.log:
            return
        folder = ROOT / "conversations"
        folder.mkdir(exist_ok=True)
        path = folder / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
        path.write_text("\n".join(f"[{t:%H:%M:%S}] {who}: {text}" for t, who, text in self.log), encoding="utf8")
        print(f"Conversation saved to {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video file to use instead of the webcam")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    Interpreter().run(args.video or 0)
