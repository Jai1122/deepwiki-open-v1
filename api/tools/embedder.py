from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from api.config import configs
import logging
import os

def get_embedder():
    embedder_config = configs["embedder"]
    provider = embedder_config.get("provider")
    model = embedder_config.get("model_kwargs", {}).get("model")

    if provider == "vllm":
        base_url = os.environ.get("VLLM_API_BASE_URL")
        api_key = os.environ.get("VLLM_API_KEY")
        return OpenAIEmbeddings(model=model, base_url=base_url, api_key=api_key)
    else:
        base_url = os.environ.get("OLLAMA_HOST")
        return OllamaEmbeddings(model=model, base_url=base_url)
