# Import necessary modules for txt file processing
from langchain_community.document_loaders import TextLoader  # Updated import for TextLoader
from langchain_openai import OpenAIEmbeddings  # Updated import path
from langchain_community.vectorstores import FAISS  # Updated import path
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Updated import path
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = "data/data.txt" 
FAISS_PATH = "data/faiss_txt"

# Load the txt file
loader = TextLoader(file_path=DATA_PATH, encoding='utf-8') 
documents = loader.load()

# Split the documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
texts = text_splitter.split_documents(documents) 

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Create the vector store from the split texts
vectorstore = FAISS.from_documents(texts, embeddings)

# Save the vector store locally for later use
vectorstore.save_local(FAISS_PATH)
