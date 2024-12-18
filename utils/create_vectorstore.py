from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data/reviews_final.csv"
FAISS_PATH = "data/faiss"

loader = CSVLoader(file_path=DATA_PATH)
documents = loader.load()

documents = [{"text": doc[0]} for doc in documents]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1028, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Create the vector store
vectorstore = FAISS.from_documents(documents, embeddings)

# Save vector store
vectorstore.save_local(FAISS_PATH)
