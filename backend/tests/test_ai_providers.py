import unittest
from unittest.mock import patch, MagicMock
import os

# Add the backend directory to the path so we can import our modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from backend.ai_providers.base_provider import AIProvider
from backend.ai_providers.azure_provider import AzureProvider
from backend.ai_providers.vllm_provider import VLLMProvider
from backend.config.manager import ConfigManager

class TestAIProviders(unittest.TestCase):
    
    def setUp(self):
        # Mock environment variables for testing
        self.mock_env_vars = {
            'AI_PROVIDER': 'azure',
            'AZURE_DEPLOYMENT_NAME': 'test-deployment',
            'AZURE_API_VERSION': '2024-02-15-preview',
            'AZURE_ENDPOINT': 'https://test-resource.openai.azure.com',
            'AZURE_API_KEY': 'test-key',
            'AZURE_EMBEDDINGS_DEPLOYMENT_NAME': 'test-embeddings'
        }
        
        # Set up environment
        for key, value in self.mock_env_vars.items():
            os.environ[key] = value
    
    def tearDown(self):
        # Clean up environment variables
        for key in self.mock_env_vars.keys():
            if key in os.environ:
                del os.environ[key]
    
    def test_base_provider_abstract(self):
        """Test that AIProvider is abstract and cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            AIProvider()
    
    def test_azure_provider_initialization(self):
        """Test AzureProvider initialization"""
        config = {
            'AZURE_DEPLOYMENT_NAME': 'test-deployment',
            'AZURE_API_VERSION': '2024-02-15-preview',
            'AZURE_ENDPOINT': 'https://test-resource.openai.azure.com',
            'AZURE_API_KEY': 'test-key',
            'AZURE_EMBEDDINGS_DEPLOYMENT_NAME': 'test-embeddings'
        }
        provider = AzureProvider(config)
        
        # Should be able to instantiate without errors
        self.assertIsNotNone(provider)
        
    def test_vllm_provider_initialization(self):
        """Test VLLMProvider initialization"""
        config = {
            'VLLM_MODEL': 'meta-llama/Llama-3.2-1B-Instruct',
            'VLLM_TENSOR_PARALLEL_SIZE': 1,
            'VLLM_EMBEDDINGS_MODEL': 'text-embedding-ada-002',
            'OPENAI_API_KEY': 'test-key'
        }
        provider = VLLMProvider(config)
        
        # Should be able to instantiate without errors
        self.assertIsNotNone(provider)
    
    def test_config_manager(self):
        """Test ConfigManager functionality"""
        config_manager = ConfigManager()
        self.assertEqual(config_manager.provider, 'azure')
        self.assertIsNotNone(config_manager.config)
        
        # Test setting provider
        config_manager.set_provider('vllm')
        self.assertEqual(config_manager.provider, 'vllm')

if __name__ == '__main__':
    unittest.main()