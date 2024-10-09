from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import LangChainProcessor
from ezprint import ezprint

# Initialize LangChainProcessor
langchain_processor = LangChainProcessor()

app = FastAPI()

def generate_session_id():
    return 3

class Query(BaseModel):
    question: str

@app.post("/generate")
async def generate_answer(query: Query):
    answer = langchain_processor.generate_answer(query.question, generate_session_id())
    return {"answer": answer}