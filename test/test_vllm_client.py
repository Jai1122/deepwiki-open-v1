import unittest
from unittest.mock import patch, MagicMock
from api.vllm_client import VLLMClient

class TestVLLMClient(unittest.TestCase):

    @patch('api.vllm_client.OpenAI')
    def test_init_sync_client(self, mock_openai):
        # Arrange
        api_key = "test_api_key"
        base_url = "http://localhost:8000"
        with patch.dict('os.environ', {'VLLM_API_KEY': api_key, 'VLLM_BASE_URL': base_url}):
            # Act
            client = VLLMClient()
            # Assert
            mock_openai.assert_called_with(api_key=api_key, base_url=base_url)

    @patch('api.vllm_client.AsyncOpenAI')
    def test_init_async_client(self, mock_async_openai):
        # Arrange
        api_key = "test_api_key"
        base_url = "http://localhost:8000"
        with patch.dict('os.environ', {'VLLM_API_KEY': api_key, 'VLLM_BASE_URL': base_url}):
            # Act
            client = VLLMClient()
            client.init_async_client()
            # Assert
            mock_async_openai.assert_called_with(api_key=api_key, base_url=base_url)

if __name__ == '__main__':
    unittest.main()
