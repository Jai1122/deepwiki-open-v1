from langchain_community.embeddings import OllamaEmbeddings
from api.config import configs
import logging
import os

def get_embedder():
    embedder_config = configs.get("embedder", {})
    provider = embedder_config.get("provider", "ollama")
    model = embedder_config.get("model_kwargs", {}).get("model")

    return OllamaEmbeddings(model=model)
