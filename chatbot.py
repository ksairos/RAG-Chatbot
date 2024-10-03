from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

class LangChainProcessor:
    def __init__(self, faiss_path="data/faiss", model_name="gpt-4o-mini"):
        self.faiss_path = faiss_path
        self.embeddings = OpenAIEmbeddings()
        self.retriever = FAISS.load_local(
            self.faiss_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        ).as_retriever()
        self.llm = ChatOpenAI(model=model_name)
        self.prompt = hub.pull("rlm/rag-prompt")
        self.chain = self._build_chain()

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        return (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def generate_answer(self, question: str) -> str:
        response = self.chain.invoke(question)
        return response