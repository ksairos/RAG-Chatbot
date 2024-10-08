from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.messages import trim_messages
from dotenv import load_dotenv

load_dotenv()

class LangChainProcessor:
    def __init__(self, model_name="gpt-4o-mini", chat_store={}):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model_name)
        self.trimmer = trim_messages(
             max_tokens=65,
             strategy="last",
             token_counter=self.llm,
             include_system=True,
             allow_partial=False,
             start_on="human",
        )
        self.prompt = hub.pull("rlm/rag-prompt")
        self.history_aware_retriever = self._create_history_aware_retriever()
        self.qa_chain = self._create_qa_chain()
        self.chain = self._build_chain()
        self.chat_store = chat_store

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.chat_store:
                self.chat_store[session_id] = InMemoryChatMessageHistory()
            return self.chat_store[session_id]

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs["context"])

    def _build_chain(self):
        return (
            {
                "context": create_retrieval_chain(self.history_aware_retriever, self.qa_chain) | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _create_history_aware_retriever(self):
        faiss_path="data/faiss"
        
        retriever = FAISS.load_local(
            faiss_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        ).as_retriever()

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
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
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


    def generate_answer(self, question: str, session_id) -> str:
        with_message_history = RunnableWithMessageHistory(
            self.chain,
            get_session_history=self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        config = {"configurable": {"session_id": session_id}}
        response = with_message_history.invoke({"input": question}, config=config)
        
        return response