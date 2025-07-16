from langchain_community.embeddings import OllamaEmbeddings
from api.vllm_client import VllmEmbeddings
from api.config import configs
import logging
import os

def get_embedder():
    embedder_config = configs.get("embedder", {})
    provider = embedder_config.get("provider", "ollama")
    model = embedder_config.get("model_kwargs", {}).get("model")

    if provider == "vllm":
        base_url = os.environ.get("VLLM_API_BASE_URL")
        api_key = os.environ.get("VLLM_API_KEY")
        embedder = VllmEmbeddings(model=model, base_url=base_url, api_key=api_key)
        logging.info(f"Embedder attributes: {dir(embedder)}")
        return embedder
    else:
        return OllamaEmbeddings(model=model)
