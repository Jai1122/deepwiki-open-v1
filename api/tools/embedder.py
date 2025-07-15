from langchain_community.embeddings import OllamaEmbeddings
from api.config import configs
import logging
import os

def get_embedder():
    embedder_config = configs["embedder"]
    model = embedder_config.get("model_kwargs", {}).get("model")
    base_url = os.environ.get("OLLAMA_HOST")

    if model:
        return OllamaEmbeddings(model=model, base_url=base_url)
    else:
        return OllamaEmbeddings(base_url=base_url)
