"""
One-time setup: load medical PDF(s) into the Pinecone "medical-chatbot" index.

Usage (from the med_chatbot folder):
    python ingest.py                        # every PDF in med_chatbot/data/ (only if the index is empty)
    python ingest.py path/to/book.pdf       # specific file(s)
    python ingest.py --add new_book.pdf     # add more books to an index that already has data

Without --add, the upload is skipped when the index already has vectors (see init_vector_store).
"""
import argparse
from pathlib import Path
from dotenv import load_dotenv
from pdf_processor import process_pdf
from vector_store import get_vector_store, init_vector_store

DATA_DIR = Path(__file__).parent / "data"


def main():
    load_dotenv(Path(__file__).parent.parent / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="*")
    parser.add_argument("--add", action="store_true", help="add to an index that already has data")
    args = parser.parse_args()

    pdfs = [Path(p) for p in args.pdfs] or sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found. Put medical PDFs in {DATA_DIR} or pass paths as arguments.")
        return

    texts = []
    for pdf in pdfs:
        print(f"Reading {pdf.name}...")
        texts.extend(process_pdf(str(pdf)))
    print(f"{len(texts)} chunks from {len(pdfs)} PDF(s)")

    if args.add:
        print("Adding to the existing index...")
        get_vector_store().add_documents(texts)
    else:
        init_vector_store(texts)
    print("Done.")


if __name__ == "__main__":
    main()
