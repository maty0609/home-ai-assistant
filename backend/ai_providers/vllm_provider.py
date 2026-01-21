from langchain_community.llms import VLLM
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from .base_provider import AIProvider

class VLLMProvider(AIProvider):
    def __init__(self, config):
        self.config = config
        self.chat_model = None
        self.embeddings = None
        self.vector_store = None
    
    def get_chat_model(self):
        if not self.chat_model:
            self.chat_model = VLLM(
                model=self.config['VLLM_MODEL'],
                tensor_parallel_size=self.config['VLLM_TENSOR_PARALLEL_SIZE'],
                # Add other vLLM specific parameters
            )
        return self.chat_model
    
    def get_embeddings(self):
        if not self.embeddings:
            self.embeddings = OpenAIEmbeddings(
                model=self.config['VLLM_EMBEDDINGS_MODEL'],
                openai_api_key=self.config['OPENAI_API_KEY']  # Using OpenAI embeddings for simplicity
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