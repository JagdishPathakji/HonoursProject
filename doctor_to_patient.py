"""
Medical interpreter, doctor -> patient.

The doctor types a reply, or presses Enter on an empty line to speak it (6 s, English or any Indian language
Sarvam supports; it is translated to English). The reply is converted to ISL gloss and played as sign videos
for the patient (see sign_player.py). Type q to quit; press q in the video window to stop playback.
"""
from pathlib import Path

import cv2
from dotenv import load_dotenv

from sign_player import SignPlayer, text_to_signs

ROOT = Path(__file__).parent
RECORD_SECONDS = 6


def listen():
    """Record the doctor and return the English transcript (translated if spoken in another Indian language)."""
    from sarvam_ai_speech import record_audio_to_base64, speech_to_text_translate
    response = speech_to_text_translate(record_audio_to_base64(duration=RECORD_SECONDS))
    if isinstance(response, dict):  # {"error": ...}
        print(f"Speech recognition failed: {response.get('error', response)}")
        return ""
    transcript, language = response[0], response[1]
    if transcript:
        print(f"Heard ({language or 'unknown language'}): {transcript}")
    return transcript or ""


def reply_to_patient(text, player):
    tokens = text_to_signs(text)
    print("Signing:", " ".join(tokens))
    player.play(tokens, caption=text)
    return tokens


def main():
    load_dotenv(ROOT / ".env")
    player = SignPlayer()
    print(__doc__)
    while True:
        text = input("\nDoctor (type, or Enter to speak, q to quit): ").strip()
        if text.lower() == "q":
            break
        if not text:
            text = listen()
            if not text:
                continue
        reply_to_patient(text, player)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
