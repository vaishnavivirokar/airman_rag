import json, faiss
from rank_bm25 import BM25Okapi
from app.embeddings import embed

index = faiss.read_index("vector_store/index.faiss")
metadata = json.load(open("vector_store/metadata.json"))

corpus = [m["text"] for m in metadata]
bm25 = BM25Okapi([c.split() for c in corpus])

def retrieve(query, k=3):
    vec = embed([query])
    D,I = index.search(vec,k)

    vector_chunks = [metadata[i] for i in I[0]]

    bm25_scores = bm25.get_scores(query.split())
    bm25_top = sorted(range(len(bm25_scores)),
                      key=lambda i:bm25_scores[i],
                      reverse=True)[:k]

    bm25_chunks = [metadata[i] for i in bm25_top]

    seen = set()
    combined = []
    for c in vector_chunks + bm25_chunks:
        key = (c.get("doc"), c.get("page"), c.get("text", "")[:50])
        if key not in seen:
            seen.add(key)
            combined.append(c)
        if len(combined) >= k:
            break
    return combined[:k]
