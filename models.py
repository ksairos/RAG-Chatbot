from sqlalchemy import create_engine, Column, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import uuid

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String)
    created_at = Column(DateTime, default=datetime.now)
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = 'messages'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey('conversations.id'))
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    conversation = relationship("Conversation", back_populates="messages")
