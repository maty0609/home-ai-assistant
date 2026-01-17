import os
import sys
import unittest
from unittest.mock import patch

# Add the backend directory to the path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import LLMConfig

class TestProviderSwitching(unittest.TestCase):
    
    def test_azure_openai_provider_initialization(self):
        """Test that Azure OpenAI provider initializes correctly"""
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
    
    def test_openai_provider_initialization(self):
        """Test that OpenAI provider initializes correctly"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'OPENAI_BASE_URL': 'https://api.openai.com/v1',
            'OPENAI_API_KEY': 'test-key'
        }):
            config = LLMConfig()
            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.openai_config['base_url'], 'https://api.openai.com/v1')
            self.assertEqual(config.openai_config['api_key'], 'test-key')
    
    def test_backward_compatibility_azure_default(self):
        """Test that default behavior remains Azure OpenAI (backward compatibility)"""
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig()
            self.assertEqual(config.provider, "azure_openai")
    
    def test_provider_selection_logic(self):
        """Test that the provider selection logic works correctly"""
        # Test Azure OpenAI
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'azure_openai',
            'AZURE_OPENAI_URL': 'https://test.azure.com',
            'AZURE_OPENAI_API_KEY': 'test-key'
        }):
            config = LLMConfig()
            self.assertEqual(config.provider, "azure_openai")
        
        # Test OpenAI
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'OPENAI_BASE_URL': 'https://api.openai.com/v1',
            'OPENAI_API_KEY': 'test-key'
        }):
            config = LLMConfig()
            self.assertEqual(config.provider, "openai")
        
        # Test invalid provider
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'invalid_provider'
        }):
            with self.assertRaises(ValueError) as context:
                LLMConfig()
            self.assertIn("Unsupported provider", str(context.exception))

if __name__ == '__main__':
    unittest.main()
