"""
OpenAI-compatible client for vLLM integration.
This client provides compatibility with vLLM's OpenAI-compatible API.
"""

import asyncio
from openai import OpenAI
from adalflow.core.types import ModelType


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
        # Store parameters for debugging
        self.api_key = api_key or "dummy"
        self.base_url = base_url
        self.kwargs = kwargs
        
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client with base_url='{base_url}', api_key={'SET' if api_key else 'NOT_SET'}: {e}")
    
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
            OpenAI API response (AsyncStreamWrapper if stream=True, otherwise response object)
        """
        if api_kwargs is None:
            api_kwargs = {}
        
        # Check if this is a streaming request
        is_streaming = api_kwargs.get("stream", False)
        
        # The sync OpenAI client doesn't return coroutines, so we need to run in executor
        def _make_sync_call():
            if "messages" in api_kwargs:
                # Chat completions
                if is_streaming:
                    # For streaming, create and return a stream wrapper
                    stream = self.client.chat.completions.create(**api_kwargs)
                    return AsyncStreamWrapper(stream)
                else:
                    # For non-streaming, make sync call
                    return self.client.chat.completions.create(**api_kwargs)
            elif "input" in api_kwargs:
                # Embeddings (embeddings don't support streaming)
                return self.client.embeddings.create(**api_kwargs)
            else:
                # Default to chat completions
                if is_streaming:
                    stream = self.client.chat.completions.create(**api_kwargs)
                    return AsyncStreamWrapper(stream)
                else:
                    return self.client.chat.completions.create(**api_kwargs)
        
        # Run the sync call in an executor to make it async
        if is_streaming:
            # For streaming, we return the wrapper directly (already handles async)
            return _make_sync_call()
        else:
            # For non-streaming, run in executor
            return await asyncio.get_event_loop().run_in_executor(None, _make_sync_call)
    
    def call(self, api_kwargs=None, model_type=None):
        """
        Make a sync call to the OpenAI API.
        
        Args:
            api_kwargs: API arguments dictionary
            model_type: Type of model (used for determining endpoint)
        
        Returns:
            OpenAI API response
        """
        if api_kwargs is None:
            api_kwargs = {}
        
        # Check if this is a streaming request
        is_streaming = api_kwargs.get("stream", False)
        
        # Determine which endpoint to call based on the presence of messages or input
        if "messages" in api_kwargs:
            # Chat completions
            if is_streaming:
                # For streaming, create and return a stream wrapper
                stream = self.client.chat.completions.create(**api_kwargs)
                return AsyncStreamWrapper(stream)
            else:
                # For non-streaming, make sync call
                return self.client.chat.completions.create(**api_kwargs)
        elif "input" in api_kwargs:
            # Embeddings (embeddings don't support streaming)
            return self.client.embeddings.create(**api_kwargs)
        else:
            # Default to chat completions
            if is_streaming:
                stream = self.client.chat.completions.create(**api_kwargs)
                return AsyncStreamWrapper(stream)
            else:
                return self.client.chat.completions.create(**api_kwargs)
    
    def chat_completion_parser(self, response):
        """
        Parse chat completion response to extract content.
        
        Args:
            response: OpenAI chat completion response
            
        Returns:
            str: The content of the response
        """
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content
            elif hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                return choice.delta.content
        return ""


class AsyncStreamWrapper:
    """Wrapper to ensure OpenAI Stream objects work properly with async iteration"""
    
    def __init__(self, stream):
        self.stream = stream
        self._is_async = hasattr(stream, '__aiter__')
        self._exhausted = False
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self._exhausted:
            raise StopAsyncIteration
            
        try:
            if self._is_async:
                # Stream is already async
                return await self.stream.__anext__()
            else:
                # Stream is sync, need to handle StopIteration properly
                import asyncio
                
                def _safe_next():
                    """Safely get next item or return sentinel"""
                    try:
                        return next(self.stream)
                    except StopIteration:
                        return StopIteration  # Return as sentinel, not raise
                
                result = await asyncio.get_event_loop().run_in_executor(None, _safe_next)
                
                if result is StopIteration:
                    self._exhausted = True
                    raise StopAsyncIteration
                    
                return result
                
        except StopAsyncIteration:
            self._exhausted = True
            raise
        except Exception as e:
            # Handle any other exceptions
            self._exhausted = True
            raise StopAsyncIteration from e