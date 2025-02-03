import os
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="ttext-embedding-3-large",  # Your deployment name
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
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path, "category": "txt"}
                )
                documents.append(doc)
    return documents

if __name__ == "__main__":
    # Load documents from a directory
    docs_directory = os.getenv("DOCUMENTS_PATH")
    documents = load_txt_files(docs_directory)
    
    # Add documents to Chroma
    vector_store = add_documents_to_chroma(documents)
    
    print("Documents added to ChromaDB successfully!")
