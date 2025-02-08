import os
import pymupdf
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="ttext-embedding-3-large",
    azure_endpoint=os.getenv("EMB_OPENAI_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-08-01-preview"
)

def add_documents_to_chroma(documents, persist_directory=os.getenv("CHROMADB_PATH")):
    """
    Add documents to ChromaDB with Azure OpenAI embeddings
    """
    # Create vector store with documents
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vector_store

def load_txt_files(directory_path):
    """
    Load all .txt files from a directory and convert them to Document objects
    """
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path, "category": "txt"}
                    )
                    documents.append(doc)
                    print("Document " + filename + " added!")
            except Exception as e:
                print(f"Error reading text file {file_path}: {e}")
    return documents

def load_pdf_files(directory_path):
    """
    Load all PDF files from a directory and convert them to Document objects using PyMuPDF.
    """
    documents = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            try:
                # Open the PDF file using pymupdf
                doc = pymupdf.open(file_path)
                text = ""
                # Extract text from each page
                for page in doc:
                    text += page.get_text() + "\n"
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "category": "pdf"}
                ))
                print("Document " + filename + " added!")
            except Exception as e:
                print(f"Error reading PDF file {file_path}: {e}")
    return documents

if __name__ == "__main__":
    docs_directory = os.getenv("DOCUMENTS_PATH")
    
    documents = []
    documents.extend(load_txt_files(docs_directory))
    documents.extend(load_pdf_files(docs_directory))

    # Add all loaded documents into ChromaDB
    vector_store = add_documents_to_chroma(documents)
    
    print("Documents added to ChromaDB successfully!")
