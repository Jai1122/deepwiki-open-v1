from langchain_openai import OpenAIEmbeddings
import logging
import json

logger = logging.getLogger(__name__)

class VllmEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts, chunk_size=0):
        logger.info(f"Sending {len(texts)} texts to VLLM for embedding.")

        # Manually construct the request body
        request_body = {
            "input": texts,
            "model": self.model,
        }

        logger.info(f"VLLM request body: {request_body}")

        try:
            # Use the raw httpx client to send the request
            response = self.client.post(
                "/embeddings",
                json=request_body,
            )

            response.raise_for_status()

            response_json = response.json()

            logger.info(f"Raw response from VLLM: {response_json}")

            if response_json and "data" in response_json and response_json["data"]:
                return [r["embedding"] for r in response_json["data"]]
            else:
                raise ValueError("VLLM response is None or has no data.")

        except Exception as e:
            logger.error(f"Error calling VLLM: {e}")
            raise e
