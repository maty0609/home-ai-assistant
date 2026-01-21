import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self):
        self.provider = os.getenv('AI_PROVIDER', 'azure')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        config = {}
        if self.provider == 'azure':
            config.update({
                'AZURE_DEPLOYMENT_NAME': os.getenv('AZURE_DEPLOYMENT_NAME'),
                'AZURE_API_VERSION': os.getenv('AZURE_API_VERSION'),
                'AZURE_ENDPOINT': os.getenv('AZURE_ENDPOINT'),
                'AZURE_API_KEY': os.getenv('AZURE_API_KEY'),
                'AZURE_EMBEDDINGS_DEPLOYMENT_NAME': os.getenv('AZURE_EMBEDDINGS_DEPLOYMENT_NAME')
            })
        elif self.provider == 'vllm':
            config.update({
                'VLLM_MODEL': os.getenv('VLLM_MODEL'),
                'VLLM_TENSOR_PARALLEL_SIZE': int(os.getenv('VLLM_TENSOR_PARALLEL_SIZE', 1)),
                'VLLM_EMBEDDINGS_MODEL': os.getenv('VLLM_EMBEDDINGS_MODEL'),
                'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY')
            })
        return config
    
    def set_provider(self, provider: str):
        self.provider = provider
        self.config = self._load_config()
    
    def get_provider_config(self):
        return self.config