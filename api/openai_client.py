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
    
    def convert_inputs_to_api_kwargs(self, input=None, model_kwargs=None, model_type=None):
        """
        Convert input and model kwargs to OpenAI API format.
        
        Args:
            input: Input text or messages
            model_kwargs: Model parameters (model, temperature, max_tokens, etc.)
            model_type: Type of model (used for compatibility, ignored for OpenAI)
        
        Returns:
            dict: API kwargs formatted for OpenAI API
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        api_kwargs = model_kwargs.copy()
        
        # Handle different input types
        if input is not None:
            if isinstance(input, str):
                # For chat completions, wrap string in messages format
                api_kwargs["messages"] = [{"role": "user", "content": input}]
            elif isinstance(input, list):
                # Already in messages format
                api_kwargs["messages"] = input
            else:
                # For other input types (embeddings, etc.)
                api_kwargs["input"] = input
        
        return api_kwargs
    
    async def acall(self, api_kwargs=None, model_type=None):
        """
        Make an async call to the OpenAI API.
        
        Args:
            api_kwargs: API arguments dictionary
            model_type: Type of model (used for determining endpoint)
        
        Returns:
            OpenAI API response (Stream object if stream=True, otherwise response object)
        """
        if api_kwargs is None:
            api_kwargs = {}
        
        # Check if this is a streaming request
        is_streaming = api_kwargs.get("stream", False)
        
        # Determine which endpoint to call based on the presence of messages or input
        if "messages" in api_kwargs:
            # Chat completions
            if is_streaming:
                # For streaming, return the stream object directly (don't await it)
                return self.client.chat.completions.create(**api_kwargs)
            else:
                # For non-streaming, await the response
                return await self.client.chat.completions.create(**api_kwargs)
        elif "input" in api_kwargs:
            # Embeddings (embeddings don't support streaming)
            return await self.client.embeddings.create(**api_kwargs)
        else:
            # Default to chat completions
            if is_streaming:
                return self.client.chat.completions.create(**api_kwargs)
            else:
                return await self.client.chat.completions.create(**api_kwargs)