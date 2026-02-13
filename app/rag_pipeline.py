from app.retriever import retrieve
from app.llm import generate,REFUSAL

def ask_question(q,debug=False):

    chunks=retrieve(q)
    context="\n".join([c["text"] for c in chunks])

    answer=generate(context,q)

    citations=[f'{c["doc"]} page {c["page"]+1}' for c in chunks]

    return {
        "answer":answer if answer else REFUSAL,
        "citations":citations,
        "chunks":chunks if debug else []
    }
