## Overview
This project (SIH 2024) aims to bridge the communication gap between the deaf community and individuals who do not know sign language by providing a robust system to convert spoken language into Indian Sign Language (ISL) and vice versa. The tool facilitates seamless interaction in various scenarios, such as healthcare and public announcements, empowering the deaf community to access critical information and engage fully in society.

## Key Features
- **Speech to ISL Conversion**:
  - Converts spoken language (English/Multiple Indic languages) into ISL gestures using real-time speech recognition and text-to-sign conversion.
  - Ensures accuracy and supports regional ISL variations.
- **Text to ISL Conversion**:
  - Displays ISL gestures corresponding to given text inputs, satisfying use cases like railway announcements.
- **ISL to Text Conversion**:
  - Recognizes ISL gestures captured through a camera and converts them into text. Currently supports letter-based gestures, with word-level gesture recognition under development.

## Setup (Windows, Python 3.12)

```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env
```

Then open `.env` and fill in `GROQ_API_KEY`, `SARVAM_AI_API_KEY` and `PINECONE_API_KEY`.

| Module | Command | Needs |
|---|---|---|
| 1. Text → ISL | `python text_isl.py` | Groq key |
| 2. Speech → ISL | `python speech_to_isl.py` | Microphone, Sarvam key |
| 3. ISL → Text | `python inference_classifier.py` | Webcam |
| 4. ISL → Speech | `python isl_to_speech.py` | Webcam, speakers |
| 5. Medical chatbot | `cd med_chatbot && python main.py` | Groq + Pinecone keys |
| **Medical interpreter (both directions, one screen)** | `python interpreter.py` | Webcam, mic, speakers, Groq + Sarvam keys |
| Interpreter: patient → doctor only | `python patient_to_doctor.py` | Webcam, speakers, Groq key |
| Interpreter: doctor → patient only | `python doctor_to_patient.py` | Mic (optional), Groq + Sarvam keys |

**Module 5 one-time setup:** put the medical PDF(s) in `med_chatbot/data/` (they are gitignored) and run `cd med_chatbot && python ingest.py` to load them into your Pinecone index.

**Patient → doctor:** the patient signs medical words one at a time (45 words: symptoms, body parts, conditions, needs, time; see `med_dataset/vocab.py`), lowering the hands between signs. Each word is shown and spoken. Enter turns the words into a sentence for the doctor using Groq (e.g. "stomach pain yesterday" → "I had stomach pain yesterday."). How the dataset and model were built is in `med_dataset/README.md`.

**Doctor → patient:** the doctor types or speaks a reply (Sarvam also translates Hindi and other Indian languages). Groq rewrites it as ISL gloss (simple root words, ISL word order), each word is matched to one of ~4,800 sign videos, words without a video are replaced by a same-meaning synonym, and anything left is finger-spelled, so the meaning is never changed. Videos download once into `sign_cache/`. Example: "Take one tablet twice a day after food" → `after food 2 day tablet 1 take`.

**`interpreter.py`** puts both directions on one screen: patient's webcam on the left, the doctor's signed reply on the right, the conversation underneath (saved to `conversations/`, which is gitignored). Keys: Enter = send the patient's signs, Tab = doctor types, m = doctor speaks, r = replay, q = quit. `--video clip.mp4` uses a recorded video instead of the webcam.

**Module 4 controls:** hold a sign steady for about half a second to accept it. Letters build a word, the SPACE sign speaks it, BACKSPACE deletes a letter, word signs (PAIN, CALL, HELP, ACCIDENT) are spoken straight away, and NEXT speaks the whole sentence. Keyboard: space / enter / backspace / c (clear) / q (quit).

Unused files from earlier approaches are in `archive/`.

## Technologies Used
- **MediaPipe**: Used for detecting and rendering hand landmarks for text-to-ISL conversion.
- **Sarvam AI**: Enables speech-to-text functionality for spoken language recognition.
- **Python**: Backend implementation for processing inputs and generating outputs.
- **OpenCV**: Used for real-time video feed capture and gesture recognition.
- **Natural Language Processing (NLP)**: Translates recognized speech into text, forming the basis for ISL generation using GenAI.

## Use Cases
1. **Public Announcements**:
   - Converts audio/text announcements at railway stations or other public venues into ISL gestures, ensuring accessibility for the deaf community.
2. **Mobile App for ISL Interpretation**:
   - Enables normal individuals to use their mobile cameras to interpret ISL gestures by deaf individuals.

## Current Status
- **Speech to ISL**:
  - Fully functional, with accurate conversion of spoken content into ISL gestures.
  - Tested and validated for use cases like public announcements.
- **ISL to Text**:
  - Successfully implemented letter-based recognition.
  - Word-level gesture recognition is in progress, with challenges related to complex training on gesture sequences.


## Challenges and Future Work
- **Word-Level Gesture Recognition**:
  - Developing a robust model for recognizing dynamic gesture sequences.
- **Regional ISL Variations**:
  - Incorporating adaptability for regional differences in ISL.

## Full-Stack Project Repository
The complete source code for the integrated full-stack application — including both frontend and backend components — is available at the following GitHub repository:

🔗 [Lingua - GitHub Repository](https://github.com/Team-GenX-H4CK3RS/Lingua)


## Acknowledgements
- **Sarvam AI** for speech recognition.
