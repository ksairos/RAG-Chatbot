# Import necessary modules for txt file processing
from langchain_community.document_loaders import TextLoader  # Updated import for TextLoader
from langchain_openai import OpenAIEmbeddings  # Updated import path
from langchain_community.vectorstores import FAISS  # Updated import path
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Updated import path
from dotenv import load_dotenv

load_dotenv()

# Define paths
DATA_PATH = "data/data.txt"  # Path to your txt file
FAISS_PATH = "data/faiss_txt"  # Path to save the FAISS index

# Load the txt file
loader = TextLoader(file_path=DATA_PATH, encoding='utf-8')  # Specify encoding if necessary
documents = loader.load()  # Returns a list of Document objects with 'page_content'

# Split the documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)  # 'texts' is a list of Document objects

# Initialize the embeddings model
embeddings = OpenAIEmbeddings()  # You may need to set your OpenAI API key

# Create the vector store from the split texts
vectorstore = FAISS.from_documents(texts, embeddings)

# Save the vector store locally for later use
vectorstore.save_local(FAISS_PATH)
