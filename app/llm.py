import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

REFUSAL = "This information is not available in the provided document(s)."

def generate(context, question):

    if not context.strip():
        return REFUSAL

    prompt=f"""
Answer ONLY using provided context.
If not found return EXACT:
{REFUSAL}

Context:
{context}

Question:
{question}
"""

    msg=client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        messages=[{"role":"user","content":prompt}]
    )

    return msg.content[0].text
