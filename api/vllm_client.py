from langchain_openai import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

class VllmEmbeddings(OpenAIEmbeddings):
    def _get_len_safe_embeddings(self, texts, *, engine, chunk_size=None):
        logger.info(f"Sending {len(texts)} texts to VLLM for embedding.")
        response = super()._get_len_safe_embeddings(texts, engine=engine, chunk_size=chunk_size)
        logger.info(f"Received {len(response)} embeddings from VLLM.")
        return response
