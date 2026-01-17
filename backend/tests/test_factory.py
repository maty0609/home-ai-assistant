import os
import sys
import unittest
from unittest.mock import patch

# Add the backend directory to the path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import LLMConfig

class TestLLMFactory(unittest.TestCase):
    
    def test_azure_config_creation(self):
        """Test that Azure configuration creates correct objects"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'azure_openai',
            'AZURE_OPENAI_URL': 'https://test.azure.com',
            'AZURE_OPENAI_API_KEY': 'test-key',
            'AZURE_DEPLOYMENT_NAME': 'gpt-4-test',
            'EMB_AZURE_DEPLOYMENT_NAME': 'text-embedding-test'
        }):
            config = LLMConfig()
            self.assertEqual(config.provider, "azure_openai")
            self.assertEqual(config.azure_config['api_base'], 'https://test.azure.com')
            self.assertEqual(config.azure_config['api_key'], 'test-key')
            self.assertEqual(config.azure_config['deployment_name'], 'gpt-4-test')
            self.assertEqual(config.azure_config['embedding_deployment_name'], 'text-embedding-test')
    
    def test_openai_config_creation(self):
        """Test that OpenAI configuration creates correct objects"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'OPENAI_BASE_URL': 'https://api.openai.com/v1',
            'OPENAI_API_KEY': 'test-key'
        }):
            config = LLMConfig()
            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.openai_config['base_url'], 'https://api.openai.com/v1')
            self.assertEqual(config.openai_config['api_key'], 'test-key')

if __name__ == '__main__':
    unittest.main()
