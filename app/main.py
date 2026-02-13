from fastapi import FastAPI
from pydantic import BaseModel
from app.ingest import ingest
from app.rag_pipeline import ask_question

app=FastAPI()

class Query(BaseModel):
    question:str
    debug:bool=False

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/ingest")
def run_ingest():
    ingest()
    return {"status":"ingested"}

@app.post("/ask")
def ask(q:Query):
    return ask_question(q.question,q.debug)
