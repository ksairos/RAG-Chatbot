from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import LangChainProcessor

# Initialize LangChainProcessor
langchain_processor = LangChainProcessor()

app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str

@app.post("/generate")
async def generate_answer(query: Query):
    answer = langchain_processor.generate_answer(query.question, query.session_id)
    return {"answer": answer}