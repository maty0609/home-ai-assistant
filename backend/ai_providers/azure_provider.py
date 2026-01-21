from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from .base_provider import AIProvider

class AzureProvider(AIProvider):
    def __init__(self, config):
        self.config = config
        self.chat_model = None
        self.embeddings = None
        self.vector_store = None
    
    def get_chat_model(self):
        if not self.chat_model:
            self.chat_model = AzureChatOpenAI(
                deployment_name=self.config['AZURE_DEPLOYMENT_NAME'],
                openai_api_version=self.config['AZURE_API_VERSION'],
                azure_endpoint=self.config['AZURE_ENDPOINT'],
                openai_api_key=self.config['AZURE_API_KEY']
            )
        return self.chat_model
    
    def get_embeddings(self):
        if not self.embeddings:
            self.embeddings = AzureOpenAIEmbeddings(
                deployment_name=self.config['AZURE_EMBEDDINGS_DEPLOYMENT_NAME'],
                openai_api_version=self.config['AZURE_API_VERSION'],
                azure_endpoint=self.config['AZURE_ENDPOINT'],
                openai_api_key=self.config['AZURE_API_KEY']
            )
        return self.embeddings
    
    def get_vector_store(self):
        if not self.vector_store:
            self.vector_store = Chroma(
                collection_name="docs",
                embedding_function=self.get_embeddings(),
                persist_directory="./chroma_db"
            )
        return self.vector_store