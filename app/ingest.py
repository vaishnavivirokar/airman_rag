import os, json, faiss
from pypdf import PdfReader
from app.chunker import chunk_text
from app.embeddings import embed

DATA_PATH = "data"
VECTOR_PATH = "vector_store"

def ingest():
    texts, metadata = [], []

    for file in os.listdir(DATA_PATH):
        if not file.lower().endswith(".pdf"):
            continue
        reader = PdfReader(os.path.join(DATA_PATH, file))
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            chunks = chunk_text(page_text)

            for chunk in chunks:
                texts.append(chunk)
                metadata.append({"doc":file,"page":i,"text":chunk})

    vectors = embed(texts)

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)

    os.makedirs(VECTOR_PATH, exist_ok=True)
    faiss.write_index(index,"vector_store/index.faiss")
    json.dump(metadata,open("vector_store/metadata.json","w"))

if __name__=="__main__":
    ingest()
