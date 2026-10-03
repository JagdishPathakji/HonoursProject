"""
End-to-end check of the live patient -> doctor pipeline without a webcam.

Plays each full reference video frame by frame through the same code the live app uses
(MediaPipe Holistic -> SignSegmenter -> MedSignRecognizer) and reports the first sign it detects.

    python med_dataset/test_live_pipeline.py              # all words
    python med_dataset/test_live_pipeline.py fever cough  # some words
    python med_dataset/test_live_pipeline.py --video my_clip.mp4

Note: the model was trained on these same videos, so this checks the plumbing (sign detection, timing,
frame rate), not accuracy on new people. Use --video with your own recordings for that.
"""
import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from vocab import WORDS  # noqa: E402
from recognizer import MedSignRecognizer  # noqa: E402
from patient_to_doctor import SignSegmenter  # noqa: E402


def run_video(path, recognizer, holistic, fps_override=None):
    """Returns a list of [(word, prob), ...] predictions, one per detected sign."""
    cap = cv2.VideoCapture(str(path))
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 25) / fps))  # drop frames to simulate a slower webcam
    segmenter = SignSegmenter(fps)
    found, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * 960 / w)))
        frames = segmenter.feed(holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if frames is not None:
            found.append(recognizer.predict(frames))
    cap.release()
    frames = segmenter.flush()
    if frames is not None:
        found.append(recognizer.predict(frames))
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("words", nargs="*")
    parser.add_argument("--video", help="test one of your own videos instead")
    parser.add_argument("--fps", type=float, help="simulate a webcam at this frame rate (e.g. 12)")
    args = parser.parse_args()

    recognizer = MedSignRecognizer()
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)

    if args.video:
        for n, top in enumerate(run_video(args.video, recognizer, holistic, args.fps), 1):
            print(f"sign {n}: " + "  ".join(f"{w} {p:.0%}" for w, p in top))
        return

    words = args.words or WORDS
    correct = 0
    for word in words:
        found = run_video(HERE / "videos" / f"{word.replace(' ', '_')}.mp4", recognizer, holistic, args.fps)
        if not found:
            print(f"{word:15} no sign detected")
            continue
        (pred, p), *alts = found[0]
        correct += pred == word
        mark = "ok " if pred == word else "XX "
        print(f"{mark}{word:15} -> {pred:15} {p:4.0%}   ({len(found)} sign(s) detected in the video)")
    print(f"\nfirst detected sign correct: {correct}/{len(words)}")


if __name__ == "__main__":
    main()
