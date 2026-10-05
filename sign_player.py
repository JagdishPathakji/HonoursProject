"""
English text -> ISL sign videos (doctor -> patient direction). Shared by doctor_to_patient.py and interpreter.py.

    tokens = text_to_signs("Take one tablet twice a day after food")   # ['tablet', '1', 'day', '2', 'food', 'after', 'take']
    player = SignPlayer()
    player.play(tokens, caption="Take one tablet twice a day after food")

1. Groq rewrites the English into ISL gloss: simple root words, ISL word order, no grammar words.
2. Every gloss word is matched to a sign video (MED_VIDEO_IDS, ~4,800 words incl. multi-word signs).
3. Words without a video: one Groq call finds simpler synonyms that have one.
4. Anything still missing is finger-spelled (A-Z); numbers become digit signs.
Videos are downloaded from Google Drive once and cached in sign_cache/.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).parent


def _load_med_dictionary():
    """Load MED_VIDEO_IDS by file path: the project has two different 'helper' packages, so importing by name is ambiguous."""
    placeholder = "https://drive.google.com/uc?export=download&id={}"
    src = (ROOT / "med_chatbot" / "helper" / "general_dictionary.py").read_text(encoding="utf8")
    return {w: placeholder.format(i) for w, i in
            re.findall(r'"([^"]+)"\s*:\s*DRIVE_LINK_PLACEHOLDER\.format\("([^"]+)"\)', src)}


MED_VIDEO_IDS = _load_med_dictionary()

CACHE_DIR = ROOT / "sign_cache"
LOCAL_DIRS = [ROOT / "med_dataset" / "videos", ROOT / "videos"]  # already-downloaded videos, checked first
GROQ_MODEL = "openai/gpt-oss-120b"

NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9", "once": "1", "twice": "2", "thrice": "3"}
MAX_PHRASE = max(len(k.split()) for k in MED_VIDEO_IDS)


def _groq_json(prompt):
    from groq import Groq
    r = Groq().chat.completions.create(model=GROQ_MODEL, max_tokens=1500, messages=[{"role": "user", "content": prompt}])
    text = r.choices[0].message.content or ""
    match = re.search(r"[\[{].*[\]}]", text, re.S)
    return json.loads(match.group(0)) if match else None


def to_gloss(text):
    """English -> list of ISL gloss words. Falls back to the plain words if Groq is unavailable."""
    prompt = (
        "Convert this doctor's sentence for a Deaf patient into Indian Sign Language (ISL) gloss.\n"
        "Rules: use simple root words (lemmatise: 'taking'->'take', 'tablets'->'tablet'); drop articles, "
        "'is/am/are/be', and other grammar words; keep negation as 'no'; use ISL order: time first, then "
        "topic/object, then verb, question words at the end; write numbers as digits; keep medical phrases "
        "like 'blood pressure' or 'blood sugar' together as one item; keep the meaning exactly, add nothing.\n"
        f'Sentence: "{text}"\n'
        'Reply with a JSON list of lowercase strings only, e.g. ["tomorrow", "blood", "test", "come"].'
    )
    try:
        gloss = _groq_json(prompt)
        if isinstance(gloss, list) and gloss:
            return [str(g).lower().strip() for g in gloss if str(g).strip()]
    except Exception as e:
        print(f"Groq unavailable ({e}); signing the words as they are")
    return re.findall(r"[a-z0-9'-]+", text.lower())


def _roots(w):
    """Simple root-word guesses for a word with no video: days->day, breathless->breathe, worried->worry."""
    out = []
    for suffix, repl in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ing", "e"), ("ied", "y"),
                         ("ed", ""), ("ed", "e"), ("d", ""), ("less", ""), ("less", "e"), ("ness", ""), ("ly", "")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            out.append(w[:len(w) - len(suffix)] + repl)
    return out


def _match(words):
    """Greedy longest match of words against sign videos. Returns (tokens, unknown words)."""
    tokens, unknown, i = [], [], 0
    while i < len(words):
        for n in range(min(MAX_PHRASE, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + n])
            if phrase in MED_VIDEO_IDS:
                tokens.append(phrase)
                i += n
                break
        else:
            w = words[i]
            root = next((r for r in _roots(w) if r in MED_VIDEO_IDS), None)
            if root:
                tokens.append(root)
            elif w in NUMBER_WORDS:
                tokens.append(NUMBER_WORDS[w])
            elif w.isdigit():
                tokens.extend(w)
            else:
                tokens.append(None)
                unknown.append(w)
            i += 1
    return tokens, unknown


def _synonyms(unknown, sentence):
    """Ask Groq for up to 3 replacements per word that keep its meaning in this sentence; returns {word: [options]}."""
    prompt = (
        f'A doctor said to a Deaf patient: "{sentence}"\n'
        f"These words from it have no Indian Sign Language video: {json.dumps(unknown)}\n"
        "For each, give up to 3 simpler common English words or short phrases that keep EXACTLY the same meaning "
        "in this sentence, best first. Be careful with medical meaning (e.g. 'high' blood pressure means 'more' or "
        "'increase', not 'tall'; 'week' can be '7 day'). If nothing keeps the meaning, give an empty list.\n"
        'Reply with a JSON object only, mapping each word to a list of options.'
    )
    try:
        result = _groq_json(prompt)
        return {k: v if isinstance(v, list) else [v] for k, v in result.items()} if isinstance(result, dict) else {}
    except Exception:
        return {}


def text_to_signs(text):
    """English sentence -> list of sign tokens (dictionary words, digits, or single letters for finger-spelling)."""
    words = []
    for g in to_gloss(text):
        # 'high-sugar' / 'blood_vessels' -> separate words, unless the whole thing has a sign (e.g. 'x-ray')
        words.extend(g.split() if g in MED_VIDEO_IDS else re.sub(r"[-_/]", " ", g).split())
    tokens, unknown = _match(words)
    replacements = _synonyms(sorted(set(unknown)), text) if unknown else {}

    out, u = [], iter(unknown)
    for t in tokens:
        if t is not None:
            out.append(t)
            continue
        word = next(u)
        for option in replacements.get(word, []):
            replaced, still_unknown = _match(str(option).lower().split())
            if replaced and not still_unknown:
                out.extend(replaced)
                break
        else:
            out.extend(c.upper() if c.isalpha() else c for c in word if c.isalnum())  # finger-spell
    return out


def text_card(token, seconds=1.2, size=(640, 360), fps=25):
    """A short video showing the word as text, used when a sign video is missing or its link is broken,
    so the word is never silently dropped (e.g. the '1' in 'take 1 tablet')."""
    p = CACHE_DIR / f"_card_{token.replace(' ', '_')}.mp4"
    if not p.exists():
        CACHE_DIR.mkdir(exist_ok=True)
        frame = np.full((size[1], size[0], 3), 250, np.uint8)
        scale = 3.0 if len(token) <= 3 else 1.6
        (tw, th), _ = cv2.getTextSize(token.upper(), cv2.FONT_HERSHEY_SIMPLEX, scale, 6)
        cv2.putText(frame, token.upper(), ((size[0] - tw) // 2, (size[1] + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 30), 6, cv2.LINE_AA)
        cv2.putText(frame, "(no sign video)", (size[0] // 2 - 80, size[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
        out = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for _ in range(int(seconds * fps)):
            out.write(frame)
        out.release()
    return p


def video_path(token):
    """Local path of the sign video for a token, downloading it to the cache if needed.
    Falls back to a text card if there is no video or it cannot be downloaded."""
    name = token.replace(" ", "_")
    for d in LOCAL_DIRS:
        p = d / f"{name}.mp4"
        if p.exists():
            return p
    p = CACHE_DIR / f"{name}.mp4"
    if p.exists() and p.stat().st_size > 0:
        return p
    if token not in MED_VIDEO_IDS:
        return text_card(token)
    try:
        r = requests.get(MED_VIDEO_IDS[token], timeout=60)
        r.raise_for_status()
        if not r.headers.get("content-type", "").startswith("video"):
            return text_card(token)
        CACHE_DIR.mkdir(exist_ok=True)
        p.write_bytes(r.content)
        return p
    except requests.RequestException as e:
        print(f"Could not download sign for '{token}' ({e}); showing it as text")
        return text_card(token)


class SignPlayer:
    """Plays sign videos one after another in an OpenCV window, with the current word and the full sentence."""

    def __init__(self, window="Doctor -> Patient (ISL)", size=(640, 360), speed=1.0):
        self.window, self.size, self.speed = window, size, speed

    def prefetch(self, tokens):
        with ThreadPoolExecutor(max_workers=6) as pool:
            return dict(zip(tokens, pool.map(video_path, tokens)))

    def play(self, tokens, caption=""):
        """Blocks until all signs are shown. Returns False if the user pressed q to stop."""
        paths = self.prefetch(list(dict.fromkeys(tokens)))
        for i, token in enumerate(tokens):
            path = paths.get(token)
            if path is None:
                print(f"No sign video for '{token}', skipping")
                continue
            cap = cv2.VideoCapture(str(path))
            delay = max(1, int(1000 / ((cap.get(cv2.CAP_PROP_FPS) or 25) * self.speed)))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, self.size)
                self._overlay(frame, tokens, i, caption)
                cv2.imshow(self.window, frame)
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    cap.release()
                    return False
            cap.release()
        return True

    @staticmethod
    def _overlay(frame, tokens, i, caption):
        W, H = frame.shape[1], frame.shape[0]
        cv2.rectangle(frame, (0, H - 60), (W, H), (0, 0, 0), -1)
        cv2.putText(frame, tokens[i].upper(), (10, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{i + 1}/{len(tokens)}", (W - 70, H - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        if caption:
            cv2.putText(frame, caption[:70], (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
