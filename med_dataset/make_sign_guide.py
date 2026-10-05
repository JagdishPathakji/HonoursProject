"""
Make a visual guide to the signs the project recognises, so users can learn them.

    python med_dataset/make_sign_guide.py

Writes sign_guide/index.html (open it in a browser) with, for every sign, a looping animation cut to just
the sign and a strip of key frames. Sections:
    - the 45 medical words (patient_to_doctor.py, interpreter.py, medical_chatbot_app.py)
    - letters, digits and word signs of the Module 3/4 model (inference_classifier.py, isl_to_speech.py)
Source: the ISL dictionary videos the project already links to (MED_VIDEO_IDS).
"""
import html
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from vocab import VOCAB  # noqa: E402
from extract_landmarks import LANDMARKS_DIR, extract  # noqa: E402
from sign_player import video_path  # noqa: E402

OUT = ROOT / "sign_guide"
GIF_H = 240
GIF_FPS = 10
N_KEYFRAMES = 4

# Module 3/4 Random Forest model (gesture_recognizer.py). H, W and HELP are in its label list but the trained
# model can never output them; NEXT, SPACE and BACKSPACE are custom control gestures with no dictionary video.
LETTER_MODEL = {
    "One-hand letters and digits": list("123456789") + list("CILOUV"),
    "Two-hand letters": list("ABDEFGJKMNPQRSTXYZ"),
    "Word signs": ["pain", "call", "accident"],
    "Listed but never recognised by the current model (retraining needed)": ["H", "W", "help"],
}

NOTES = {
    "1": "Index finger pointing up, palm facing forward. (The dictionary video link for 1 is broken.)",
    "H": "The model has no output for this sign.",
    "W": "The model has no output for this sign.",
    "help": "The letter model has no output for HELP; the medical model does recognise 'help'.",
}

TIPS = {
    "body": "Point to or touch the body part as shown.",
}


def crop_signer(frame):
    """Dictionary videos have the signer in the centre with pictures/logos at the sides: keep the centre."""
    h, w = frame.shape[:2]
    cw = int(h * 0.95)
    if w <= cw:  # portrait full-body video (the letter/digit clips): zoom to head-to-waist so the hand shape is visible
        return frame[:int(h * 0.62), int(w * 0.12):int(w * 0.88)]
    x = (w - cw) // 2
    return frame[:, x:x + cw]


def segment_for(token, path, holistic):
    """Frames [start, end) of the sign itself: from saved landmarks if available, else detect them now."""
    saved = LANDMARKS_DIR / f"{token.replace(' ', '_')}.npz"
    if saved.exists():
        return tuple(np.load(saved)["segment"])
    return tuple(extract(path, holistic)["segment"])


def make_media(token, path, segment, name):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    s, e = segment
    step = max(1, round(fps / GIF_FPS))
    frames = []
    for i in range(e):
        ok, frame = cap.read()
        if not ok:
            break
        if i >= s and (i - s) % step == 0:
            frame = crop_signer(frame)
            frame = cv2.resize(frame, (int(frame.shape[1] * GIF_H / frame.shape[0]), GIF_H))
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        return None, None

    gif = OUT / "gifs" / f"{name}.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(1000 / GIF_FPS), loop=0, optimize=True)

    idx = np.linspace(0, len(frames) - 1, N_KEYFRAMES).astype(int)
    strip = Image.new("RGB", (frames[0].width * N_KEYFRAMES, GIF_H), "white")
    for k, i in enumerate(idx):
        strip.paste(frames[i], (k * frames[0].width, 0))
    strip = strip.resize((strip.width // 2, GIF_H // 2))
    png = OUT / "frames" / f"{name}.jpg"
    strip.save(png, quality=85)
    return gif.relative_to(OUT).as_posix(), png.relative_to(OUT).as_posix()


def card(token, gif, strip, note=""):
    label = token.upper() if len(token) == 1 else token
    note_html = f'<p class="note">{html.escape(note)}</p>' if note else ""
    if gif is None:
        return f'<div class="card missing"><h3>{html.escape(label)}</h3><p class="note">No video available</p></div>'
    return (f'<div class="card"><h3>{html.escape(label)}</h3>'
            f'<img class="anim" src="{gif}" alt="Animated ISL sign for {html.escape(label)}" loading="lazy">'
            f'<img class="strip" src="{strip}" alt="Key frames of the sign for {html.escape(label)}" loading="lazy">'
            f'{note_html}</div>')


def main():
    (OUT / "gifs").mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(exist_ok=True)
    holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)

    sections = [(f"Medical words: {group}", words, TIPS.get(group, "")) for group, words in VOCAB.items()]
    letter_sections = [(f"Letter model: {title}", tokens, "") for title, tokens in LETTER_MODEL.items()]

    def build(section_list, prefix):
        out = []
        for title, tokens, tip in section_list:
            cards = []
            for token in tokens:
                path = video_path(token)
                gif = strip = None
                if path:
                    gif, strip = make_media(token, path, segment_for(token, path, holistic),
                                            f"{prefix}_{token.replace(' ', '_')}")
                cards.append(card(token, gif, strip, NOTES.get(token, "") if "Letter" in title else ""))
                print(f"{title[:30]:30} {token:15} {'ok' if gif else 'MISSING'}")
            tip_html = f'<p class="tip">{html.escape(tip)}</p>' if tip else ""
            out.append(f'<section><h2>{html.escape(title)}</h2>{tip_html}<div class="grid">{"".join(cards)}</div></section>')
        return out

    body = build(sections, "med") + build(letter_sections, "letter")
    holistic.close()

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ISL Sign Guide</title>
<style>
:root {{ --bg:#f6f5f2; --card:#fff; --ink:#1d1d1f; --muted:#5f6368; --accent:#0b6e4f; --line:#e2e0db; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17181a; --card:#222326; --ink:#ececec; --muted:#a0a4a8; --accent:#4cc9a0; --line:#34363a; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:24px 16px 48px; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:1200px; margin:0 auto; }}
h1 {{ margin:0 0 4px; font-size:28px; }}
.lead {{ color:var(--muted); margin:0 0 8px; max-width:760px; }}
h2 {{ margin:36px 0 4px; font-size:20px; color:var(--accent); }}
.tip {{ margin:0 0 12px; color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(230px, 1fr)); gap:14px; margin-top:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px; break-inside:avoid; }}
.card h3 {{ margin:0 0 8px; font-size:17px; text-transform:capitalize; }}
.card img {{ display:block; width:100%; height:auto; border-radius:6px; background:#ccc; }}
.card .strip {{ margin-top:6px; }}
.note {{ color:var(--muted); font-size:13px; margin:6px 0 0; }}
.missing {{ opacity:.6; }}
ul {{ margin:8px 0 0; padding-left:20px; color:var(--muted); }}
@media print {{ body {{ background:#fff; color:#000; }} .anim {{ display:none; }} .grid {{ grid-template-columns:repeat(3, 1fr); }} }}
</style></head>
<body><main>
<h1>ISL Sign Guide</h1>
<p class="lead">Every sign the project recognises. The animation loops just the sign; the strip below shows its key moments
from start to end (it is also what prints). Source: Indian Sign Language dictionary videos.</p>
<ul>
<li><b>Medical words</b> (interpreter / chatbot): face the camera with your upper body in view, make one sign,
then <b>lower both hands to your sides</b> before the next one. Press Enter to send the words.</li>
<li><b>Letter model</b> (ISL → Text / Speech): hold the hand shape <b>still</b> for about half a second until the bar fills.
NEXT, SPACE and BACKSPACE are custom gestures from the training data with no dictionary video; you can use the
keyboard instead (space, enter, backspace).</li>
<li>The model learned each sign from one dictionary signer, so copy the movement and hand shape closely.</li>
</ul>
{"".join(body)}
</main></body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf8")
    print(f"\nOpen {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
