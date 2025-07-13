import os
from openai import OpenAI

class VLLMClient:
    """
    A client for interacting with a VLLM model.
    """

    def __init__(self, model: str, base_url: str, api_key: str):
        """
        Initializes the VLLM client.

        Args:
            model (str): The name of the VLLM model to use.
            base_url (str): The base URL of the VLLM API.
            api_key (str): The API key for the VLLM API.
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, prompts: list[str]) -> list[str]:
        """
        Generates text from a list of prompts.

        Args:
            prompts (list[str]): A list of prompts to generate text from.

        Returns:
            list[str]: A list of generated texts.
        """
        responses = []
        for prompt in prompts:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7,
            )
            responses.append(response.choices[0].text)
        return responses
