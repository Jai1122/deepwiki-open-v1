"""
OpenAI-compatible client for vLLM integration.
This client provides compatibility with vLLM's OpenAI-compatible API.
"""

from openai import OpenAI


class OpenAIClient:
    """OpenAI-compatible client for vLLM."""
    
    def __init__(self, api_key: str = None, base_url: str = None, **kwargs):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the API
            **kwargs: Additional arguments
        """
        self.client = OpenAI(
            api_key=api_key or "dummy",  # vLLM may not require a real API key
            base_url=base_url,
            **kwargs
        )
    
    def chat(self, messages, **kwargs):
        """
        Generate chat completions.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (model, temperature, etc.)
            
        Returns:
            OpenAI chat completion response
        """
        return self.client.chat.completions.create(
            messages=messages,
            **kwargs
        )
    
    def embeddings(self, input, **kwargs):
        """
        Generate embeddings.
        
        Args:
            input: Text input for embedding generation
            **kwargs: Additional parameters
            
        Returns:
            OpenAI embeddings response
        """
        return self.client.embeddings.create(
            input=input,
            **kwargs
        )