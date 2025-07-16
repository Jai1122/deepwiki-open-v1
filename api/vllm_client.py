from langchain_openai import OpenAIEmbeddings
import logging
import httpx
import os

logger = logging.getLogger(__name__)

class VllmEmbeddings(OpenAIEmbeddings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = kwargs.get("api_key")

    def embed_documents(self, texts, chunk_size=0):
        logger.info(f"Sending {len(texts)} texts to VLLM for embedding.")

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        json_data = {
            "input": texts,
            "model": self.model,
        }

        try:
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=json_data,
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
