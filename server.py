from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from chatbot import LangChainProcessor
from models import Base, Conversation, Message
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import uuid

# Initialize LangChainProcessor
langchain_processor = LangChainProcessor()

app = FastAPI()

DATABASE_URL = "sqlite:///./chatbot.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Query(BaseModel):
    question: str
    conversation_id: str

@app.post("/generate")
async def generate_answer(query: Query, db: Session = Depends(get_db)):
    try:
        uuid.UUID(query.conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID format")
    # Get conversation
    conversation = db.query(Conversation).filter(Conversation.id == query.conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Generate answer using LangChainProcessor
    answer = langchain_processor.generate_answer(query.question, conversation.id, db)

    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role='user',
        content=query.question
    )
    db.add(user_message)

    # Save assistant message
    assistant_message = Message(
        conversation_id=conversation.id,
        role='assistant',
        content=answer
    )
    db.add(assistant_message)

    db.commit()

    return {"answer": answer}

class CreateConversation(BaseModel):
    name: str = None

@app.post("/conversations")
async def create_conversation(conversation: CreateConversation, db: Session = Depends(get_db)):
    new_conversation = Conversation(name=conversation.name)
    db.add(new_conversation)
    db.commit()
    db.refresh(new_conversation)
    return {"conversation_id": new_conversation.id}

@app.get("/conversations")
async def get_conversations(db: Session = Depends(get_db)):
    conversations = db.query(Conversation).all()
    result = []
    for conv in conversations:
        result.append({
            "conversation_id": conv.id,
            "name": conv.name or f"Conversation {conv.id}",
            "created_at": conv.created_at.isoformat()
        })
    return result

@app.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str, db: Session = Depends(get_db)):
    # Validate UUID format
    try:
        uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID format")

    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at).all()
    result = []
    for msg in messages:
        result.append({
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat()
        })
    return result
