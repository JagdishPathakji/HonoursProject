"""
Medical RAG chatbot for a Deaf patient, two-way: ISL / speech / text in, ISL / speech / text out.

    python medical_chatbot_app.py
    python medical_chatbot_app.py --video clip.mp4   # recorded video instead of the webcam

Ask a question in any of three ways:
    sign   - sign medical words to the webcam (one at a time, lowering the hands between signs), then Enter
    type   - Tab, type the question, Enter
    speak  - m, then speak for 6 s (English, Hindi or another Indian language)
The answer comes only from the medical book in Pinecone (see med_chatbot/ingest.py). It is shown as text,
spoken aloud, and signed in the right pane. r replays the last answer, q quits.

This is guidance from a reference book, not a diagnosis.
"""
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "med_chatbot"))  # first, so 'helper' resolves to med_chatbot/helper for chatbot.py
from chatbot import init_chatbot  # noqa: E402
from vector_store import get_vector_store  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from interpreter import Interpreter  # noqa: E402
from patient_to_doctor import words_to_sentence  # noqa: E402
from sign_player import text_to_signs, video_path  # noqa: E402

ANSWER_PROMPT = (
    "You are a careful medical assistant answering a Deaf patient. The answer will be shown as text, "
    "spoken aloud, and translated into Indian Sign Language, so: answer in at most 2 short, simple sentences, "
    "plain everyday English, no jargon, no lists or symbols. "
    "Use ONLY the medical context below. If the context does not answer the question, say "
    "\"I don't know. Please see a doctor.\" Never invent facts or doses. "
    "If the symptoms could be serious (chest pain, trouble breathing, heavy bleeding, fainting), "
    "tell the patient to see a doctor now."
    "\n\n{context}"
)
HISTORY_TURNS = 4  # question/answer pairs kept for follow-up questions


class MedicalChatbotApp(Interpreter):
    TITLE = "ISL Medical Chatbot"
    REPLY_LABEL = "Chatbot"
    HINT = "Sign + Enter | Tab: type question | m: speak question | r: replay answer | q: quit"

    def __init__(self):
        super().__init__()
        print("Connecting to the medical knowledge base...")
        self.chain = init_chatbot(get_vector_store(), qa_system_prompt=ANSWER_PROMPT)
        self.history = []

    def ask(self, question, source):
        """Background thread: question -> RAG answer -> ISL signs. Posts events for the UI thread."""
        self.events.put(("question", (question, source)))
        self.events.put(("status", "Searching the medical book..."))
        result = self.chain.invoke({"input": question, "chat_history": self.history[-2 * HISTORY_TURNS:]})
        answer = result["answer"].strip()
        self.history += [HumanMessage(content=question), AIMessage(content=answer)]

        self.events.put(("status", "Translating the answer to ISL..."))
        tokens = text_to_signs(answer)
        self.events.put(("status", f"Loading {len(set(tokens))} sign videos..."))
        paths = {t: video_path(t) for t in dict.fromkeys(tokens)}
        self.events.put(("answer", (answer, tokens, paths)))

    # Interpreter hooks: signed words, typed text and speech all become questions to the chatbot
    def _patient_sentence(self, words):
        self.events.put(("status", "Understanding your signs..."))
        self.ask(words_to_sentence(words), f"signed: {' '.join(words)}")

    def _doctor_reply(self, text):
        self.ask(text, "typed/spoken")

    def handle_events(self):
        pending = []
        while not self.events.empty():
            kind, payload = self.events.get()
            if kind == "question":
                question, source = payload
                self.log.append((datetime.now(), "Patient", f"{question}   [{source}]"))
            elif kind == "answer":
                answer, tokens, paths = payload
                self.log.append((datetime.now(), self.REPLY_LABEL, f"{answer}   [signed: {' '.join(tokens)}]"))
                self.pane.start(tokens, paths, answer)
                self.speaker.say(answer)
                self.status = ""
            else:
                pending.append((kind, payload))
        for item in pending:  # status messages etc. are handled by the interpreter
            self.events.put(item)
        super().handle_events()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video file to use instead of the webcam")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    MedicalChatbotApp().run(args.video or 0)
