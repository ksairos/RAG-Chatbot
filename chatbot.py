from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, retrieval
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import trim_messages
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from models import Message

load_dotenv()

class LangChainProcessor:
    def __init__(self, model_name="gpt-4o-mini", chat_store={}):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model=model_name)
        # self.trimmer = trim_messages(
        #      max_tokens=65,
        #      strategy="last",
        #      token_counter=self.llm,
        #      include_system=True,
        #      allow_partial=False,
        #      start_on="human",
        # )
        self.history_aware_retriever = self._create_history_aware_retriever()
        self.qa_chain = self._create_qa_chain()
        self.chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)
        self.chat_store = chat_store

    def _get_session_history_db(self, session_id: str, db: Session) -> BaseChatMessageHistory:
        conversation_id = session_id
        # Retrieve messages from the database
        messages = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.created_at).all()
        chat_history = InMemoryChatMessageHistory()
        for msg in messages:
            if msg.role == 'user':
                chat_history.add_user_message(msg.content)
            elif msg.role == 'assistant':
                chat_history.add_ai_message(msg.content)
        return chat_history

    # def _format_docs(self, docs):
    #     return "\n\n".join(doc.page_content for doc in docs["context"])
    
    def _create_history_aware_retriever(self):
        faiss_path="data/faiss_txt"
        
        retriever = FAISS.load_local(
            faiss_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        ).as_retriever(search_kwargs={"k": 5}) # Modify the top k (top_k) results retrieved for the context

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        return history_aware_retriever
    
    def _create_qa_chain(self):
        system_prompt = (
            "You are a highly knowledgeable assistant with access to a large database of accurate and up-to-date information."
            "Always base your answers on the retrieved documents below and avoid speculating or providing information not found in the database."
            "If a query is out of scope or no relevant information is retrieved, politely explain that you cannot provide the answer,"
            "and guide the user to ask something within your area of knowledge."
            "\n\n"
            "Relevant information retrieved from the database: \n"
            "{context}"
            "\n"
        )
         
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        return qa_chain


    def generate_answer(self, question: str, session_id, db) -> str:
        with_message_history = RunnableWithMessageHistory(
            self.chain,
            get_session_history=lambda sid: self._get_session_history_db(sid, db),
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        config = {"configurable": {"session_id": session_id}}
        response = with_message_history.invoke({"input": question}, config=config)["answer"]
        
        return response