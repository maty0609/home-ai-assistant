# import os
# from langchain_community.vectorstores import Chroma
# from langchain_openai import AzureOpenAIEmbeddings
# from dotenv import load_dotenv
# from langchain_core.documents import Document

# load_dotenv()

# # Initialize Azure OpenAI Embeddings
# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment="ttext-embedding-3-large",
#     azure_endpoint=os.getenv("EMB_OPENAI_URL"),
#     api_key=os.getenv("OPENAI_API_KEY"),
#     api_version="2024-08-01-preview"
# )



# def add_documents_to_chroma(documents, persist_directory=os.getenv("CHROMADB_PATH")):
#     """
#     Add documents to ChromaDB with Azure OpenAI embeddings
#     """
#     # Create vector store with documents
#     vector_store = Chroma.from_documents(
#         documents=documents,
#         embedding=embeddings,
#         persist_directory=persist_directory
#     )

#     return vector_store

# def load_txt_files(directory_path):
#     """
#     Load all .txt files from a directory and convert them to Document objects
#     """
#     documents = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(directory_path, filename)
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as file:
#                     content = file.read()
#                     doc = Document(
#                         page_content=content,
#                         metadata={"source": file_path, "category": "txt"}
#                     )
#                     documents.append(doc)
#                     print("Document " + filename + " added!")
#             except Exception as e:
#                 print(f"Error reading text file {file_path}: {e}")
#     return documents

# def load_pdf_files(directory_path):
#     """
#     Load all PDF files from a directory using LlamaIndex's PDFReader.
#     This reader (from llama_index.readers.file) leverages PyMuPDF for optimal PDF parsing.
#     """
#     documents = []
#     try:
#         from llama_index.readers.file import PyMuPDFReader
#     except Exception as e:
#         print(f"Error importing PDFReader from llama_index.readers.file: {e}")
#         return documents

#     reader = PyMuPDFReader()
    
#     for filename in os.listdir(directory_path):
#         if filename.lower().endswith('.pdf'):
#             file_path = os.path.join(directory_path, filename)
#             try:
#                 # Load data using LlamaIndex's PDFReader
#                 loaded_docs = reader.load_data(file_path=file_path)
#                 for doc in loaded_docs:
#                     # Ensure document metadata is present.
#                     if not hasattr(doc, "metadata") or not doc.metadata:
#                         doc.metadata = {}
#                     doc.metadata["source"] = file_path
#                     doc.metadata["category"] = "pdf"
#                     documents.append(Document(page_content=doc.text, metadata=doc.metadata))
#                 print("Document " + filename + " added!")
#             except Exception as e:
#                 print(f"Error reading PDF file {file_path}: {e}")
#     return documents

# if __name__ == "__main__":
#     docs_directory = os.getenv("DOCUMENTS_PATH")
    
#     documents = []
#     documents.extend(load_txt_files(docs_directory))
#     documents.extend(load_pdf_files(docs_directory))

#     # Add all loaded documents into ChromaDB
#     vector_store = add_documents_to_chroma(documents)
    
#     print("Documents added to ChromaDB successfully!")

import chromadb
from chromadb.utils import embedding_functions
import os
import warnings
import sentence_transformers
import ssl


warnings.filterwarnings('ignore')

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

ssl_context = ssl._create_unverified_context()
#sentence_transformers.models.Transformer.requests_session.verify = False

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="Huffon/sentence-klue-roberta-base"  # You can use any HuggingFace model here
)

# Create or get a collection
chroma_client = chromadb.PersistentClient(path="/Users/matyasprokop/Documents/Git/home-ai-assistant/chroma_db")
chroma_collection = chroma_client.get_or_create_collection("example_collection",embedding_function=embedding_function)

# Add documents with metadata
chroma_collection.add(
    documents=["Liz is a good person"],
    ids=["id1"],
    metadatas=[{"category": "blog", "author": "john"}]
)
