"""Read every PDF in the data folder, split it into overlapping chunks, and store them in Chroma."""
import glob
import os

from pypdf import PdfReader

from rag import get_collection

DATA_DIR = "data"
CHUNK_SIZE = 1000     # characters per chunk
CHUNK_OVERLAP = 200   # overlap so sentences aren't cut off between chunks
BATCH = 100


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())  # collapse extra whitespace
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def doc_label(file_name: str) -> str:
    """'apple-10k.pdf' -> 'apple 10k', so the company name is searchable in every chunk."""
    return os.path.splitext(file_name)[0].replace("-", " ").replace("_", " ")


def index_file(path: str) -> tuple[int, int]:
    """Index one PDF (replacing any old chunks for it). Returns (pages, chunks)."""
    collection = get_collection()
    name = os.path.basename(path)
    reader = PdfReader(path)
    collection.delete(where={"source": name})  # so re-indexing doesn't leave stale chunks
    ids, docs, metas = [], [], []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for i, chunk in enumerate(chunk_text(text)):
            if len(chunk.strip()) < 50:  # skip near-empty chunks
                continue
            ids.append(f"{name}-p{page_no}-c{i}")
            docs.append(f"[Document: {doc_label(name)}] {chunk}")
            metas.append({"source": name, "page": page_no})
    for b in range(0, len(ids), BATCH):
        collection.upsert(ids=ids[b:b + BATCH], documents=docs[b:b + BATCH], metadatas=metas[b:b + BATCH])
    return len(reader.pages), len(ids)


def main() -> int:
    """Index every PDF in the data folder. Returns the number of chunks added."""
    pdfs = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdfs:
        print(f"No PDFs found. Put some PDF files in the '{DATA_DIR}' folder and run again.")
        return 0

    total = 0
    for path in pdfs:
        pages, n = index_file(path)
        print(f"{os.path.basename(path)}: {pages} pages -> {n} chunks")
        if not n:
            print("   WARNING: no text found. It may be a scanned/image PDF. Try a different copy.")
        total += n

    print(f"\nDone. {total} chunks added. Collection now holds {get_collection().count()} chunks.")
    return total


if __name__ == "__main__":
    main()
