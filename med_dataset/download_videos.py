"""
Step 1: download the reference sign video for every word in vocab.py.

    python med_dataset/download_videos.py

Saves med_dataset/videos/<word>.mp4 (gitignored). Already-downloaded words are skipped.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from vocab import WORDS  # noqa: E402
from sign_player import MED_VIDEO_IDS  # noqa: E402  (loaded by file path; the project has two 'helper' packages)

VIDEOS_DIR = HERE / "videos"


def download(word):
    out = VIDEOS_DIR / f"{word.replace(' ', '_')}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return word, "exists"
    try:
        r = requests.get(MED_VIDEO_IDS[word], timeout=60)
        r.raise_for_status()
        if not r.headers.get("content-type", "").startswith("video"):
            return word, f"FAILED (got {r.headers.get('content-type')}, not a video)"
        out.write_bytes(r.content)
        return word, f"ok ({len(r.content) // 1024} KB)"
    except requests.RequestException as e:
        return word, f"FAILED ({e})"


def main():
    VIDEOS_DIR.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(download, WORDS))
    for word, status in results:
        print(f"{word:15} {status}")
    failed = [w for w, s in results if s.startswith("FAILED")]
    print(f"\n{len(WORDS) - len(failed)}/{len(WORDS)} videos available in {VIDEOS_DIR}")
    if failed:
        print("Failed:", failed)


if __name__ == "__main__":
    main()
