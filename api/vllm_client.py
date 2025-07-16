from langchain_openai import OpenAIEmbeddings
import logging
import httpx
import os

logger = logging.getLogger(__name__)

class VllmEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts, chunk_size=0):
        logger.info(f"Sending {len(texts)} texts to VLLM for embedding.")

        headers = {
            "Content-Type": "application/json",
        }
        if self.openai_api_key:
            headers["Authorization"] = f"Bearer {self.openai_api_key}"

        json_data = {
            "input": texts,
            "model": self.model,
        }

        try:
            logger.info(f"Request headers: {headers}")
            logger.info(f"Request json: {json_data}")
            with httpx.Client() as client:
                response = client.post(
                    f"{self.openai_api_base}/embeddings",
                    headers=headers,
                    json=json_data,
                )

            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response content: {response.text}")
            response.raise_for_status()

            response_json = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Error calling VLLM: {e}")
            raise ValueError(f"The VLLM server returned an error. Please check the VLLM server logs for more information. Error: {e}")

            logger.info(f"Raw response from VLLM: {response_json}")

            if response_json and "data" in response_json and response_json["data"]:
                return [r["embedding"] for r in response_json["data"]]
            else:
                raise ValueError("VLLM response is None or has no data.")

        except Exception as e:
            logger.error(f"Error calling VLLM: {e}")
            raise e
