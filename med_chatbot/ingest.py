"""
One-time setup: load medical PDF(s) into the Pinecone "medical-chatbot" index.

Usage (from the med_chatbot folder):
    python ingest.py                  # every PDF in med_chatbot/data/
    python ingest.py path/to/book.pdf # specific file(s)

Skips the upload if the index already has vectors (see init_vector_store).
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
from pdf_processor import process_pdf
from vector_store import init_vector_store

DATA_DIR = Path(__file__).parent / "data"


def main():
    load_dotenv(Path(__file__).parent.parent / ".env")

    pdfs = [Path(p) for p in sys.argv[1:]] or sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found. Put medical PDFs in {DATA_DIR} or pass paths as arguments.")
        return

    texts = []
    for pdf in pdfs:
        print(f"Reading {pdf.name}...")
        texts.extend(process_pdf(str(pdf)))
    print(f"{len(texts)} chunks from {len(pdfs)} PDF(s)")

    init_vector_store(texts)
    print("Done.")


if __name__ == "__main__":
    main()
