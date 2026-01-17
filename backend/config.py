import os
from typing import Dict, Optional

class LLMConfig:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "azure_openai")
        
        # Azure OpenAI configuration (backward compatible)
        self.azure_config = {
            "api_base": os.getenv("AZURE_OPENAI_URL"),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "deployment_name": os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4"),
            "embedding_deployment_name": os.getenv("EMB_AZURE_DEPLOYMENT_NAME", "text-embedding-3-large")
        }
        
        # OpenAI configuration
        self.openai_config = {
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "api_key": os.getenv("OPENAI_API_KEY")
        }
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required configuration is provided for the selected provider."""
        if self.provider == "azure_openai":
            if not self.azure_config["api_base"]:
                raise ValueError("AZURE_OPENAI_URL is required for Azure OpenAI provider")
            if not self.azure_config["api_key"]:
                raise ValueError("AZURE_OPENAI_API_KEY is required for Azure OpenAI provider")
        elif self.provider == "openai":
            if not self.openai_config["api_key"]:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            if not self.openai_config["base_url"]:
                raise ValueError("OPENAI_BASE_URL is required for OpenAI provider")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
