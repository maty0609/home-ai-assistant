from abc import ABC, abstractmethod
from typing import List, Dict, Any

class AIProvider(ABC):
    @abstractmethod
    def get_chat_model(self):
        pass
    
    @abstractmethod
    def get_embeddings(self):
        pass
    
    @abstractmethod
    def get_vector_store(self):
        pass