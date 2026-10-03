"""
Medical vocabulary for patient -> doctor sign recognition.

Every word here has a reference sign video in med_chatbot/helper/general_dictionary.py (MED_VIDEO_IDS).
Doctor -> patient replies are not limited to this list; they can use all MED_VIDEO_IDS videos.
"""

VOCAB = {
    "symptom": [
        "pain", "fever", "cough", "cold", "headache", "vomiting", "dizzy", "breathe",
        "bleeding", "swelling", "burn", "diarrhea", "constipation", "weak", "sweat", "hurt", "injury",
    ],
    "body": [
        "head", "stomach", "throat", "ear", "eye", "tooth", "back", "arm", "foot", "heart", "neck", "knee",
    ],
    "condition": ["diabetes", "blood pressure", "asthma", "pregnancy"],
    "need": ["doctor", "medicine", "help", "emergency", "yes", "no", "sleep", "eat"],
    "time": ["today", "yesterday", "morning", "night"],
}

WORDS = [w for group in VOCAB.values() for w in group]
