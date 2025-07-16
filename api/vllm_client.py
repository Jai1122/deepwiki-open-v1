from langchain_openai import OpenAIEmbeddings
import logging

logger = logging.getLogger(__name__)

class VllmEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts, chunk_size=0):
        logger.info(f"Sending {len(texts)} texts to VLLM for embedding.")
        logger.info(f"VLLM request body: {texts}")
        try:
            response = self.client.create(input=texts, model=self.model)
            logger.info(f"Raw response from VLLM: {response}")
            if response and response.data:
                return [r.embedding for r in response.data]
            else:
                logger.error("VLLM response is None or has no data.")
                return []
        except Exception as e:
            logger.error(f"Error calling VLLM: {e}")
            return []
