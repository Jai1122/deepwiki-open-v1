from langchain_community.embeddings import OllamaEmbeddings
from api.config import configs
import logging

def get_embedder():
    embedder_config = configs["embedder"]
    model = embedder_config.get("model_kwargs", {}).get("model")

    if model:
        return OllamaEmbeddings(model=model)
    else:
        return OllamaEmbeddings()
