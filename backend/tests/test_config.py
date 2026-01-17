import os
import sys
import unittest
from unittest.mock import patch

# Add the backend directory to the path so we can import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import LLMConfig

class TestLLMConfig(unittest.TestCase):
    
    def test_default_provider_azure_openai(self):
        """Test that default provider is azure_openai"""
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig()
            self.assertEqual(config.provider, "azure_openai")
    
    def test_explicit_azure_provider(self):
        """Test explicit azure_openai provider"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'azure_openai',
            'AZURE_OPENAI_URL': 'https://test.azure.com',
            'AZURE_OPENAI_API_KEY': 'test-key'
        }):
            config = LLMConfig()
            self.assertEqual(config.provider, "azure_openai")
            self.assertEqual(config.azure_config['api_base'], 'https://test.azure.com')
            self.assertEqual(config.azure_config['api_key'], 'test-key')
    
    def test_explicit_openai_provider(self):
        """Test explicit openai provider"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'OPENAI_BASE_URL': 'https://api.openai.com/v1',
            'OPENAI_API_KEY': 'test-key'
        }):
            config = LLMConfig()
            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.openai_config['base_url'], 'https://api.openai.com/v1')
            self.assertEqual(config.openai_config['api_key'], 'test-key')
    
    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises ValueError"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'invalid_provider'
        }):
            with self.assertRaises(ValueError) as context:
                LLMConfig()
            self.assertIn("Unsupported provider", str(context.exception))
    
    def test_azure_config_validation_missing_url(self):
        """Test that missing Azure URL raises error"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'azure_openai',
            'AZURE_OPENAI_API_KEY': 'test-key'
        }):
            with self.assertRaises(ValueError) as context:
                LLMConfig()
            self.assertIn("AZURE_OPENAI_URL is required", str(context.exception))
    
    def test_azure_config_validation_missing_api_key(self):
        """Test that missing Azure API key raises error"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'azure_openai',
            'AZURE_OPENAI_URL': 'https://test.azure.com'
        }):
            with self.assertRaises(ValueError) as context:
                LLMConfig()
            self.assertIn("AZURE_OPENAI_API_KEY is required", str(context.exception))
    
    def test_openai_config_validation_missing_api_key(self):
        """Test that missing OpenAI API key raises error"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'OPENAI_BASE_URL': 'https://api.openai.com/v1'
        }):
            with self.assertRaises(ValueError) as context:
                LLMConfig()
            self.assertIn("OPENAI_API_KEY is required", str(context.exception))
    
    def test_openai_config_validation_missing_base_url(self):
        """Test that missing OpenAI base URL raises error"""
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'test-key'
        }):
            with self.assertRaises(ValueError) as context:
                LLMConfig()
            self.assertIn("OPENAI_BASE_URL is required", str(context.exception))

if __name__ == '__main__':
    unittest.main()
