import logging
from typing import List, Optional
from adalflow.core.types import Document
from api.data_pipeline import count_tokens, get_tokenizer

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages the prompt assembly by intelligently selecting, truncating, and formatting
    all components to fit within a specified token limit.
    """

    def __init__(self, model_provider: str):
        """
        Initializes the ContextManager.

        Args:
            model_provider (str): The name of the model provider (e.g., 'ollama', 'openai'),
                                  which can affect token counting.
        """
        self.is_ollama = model_provider == "ollama"

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """A helper to truncate text to a max token count."""
        if not text or max_tokens <= 0:
            return ""

        encoding = get_tokenizer(self.is_ollama)
        tokens = encoding.encode(text)

        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens)

        return text

    def build_prompt(
        self,
        system_prompt: str,
        query: str,
        conversation_history: str,
        file_content: Optional[str],
        file_path: Optional[str],
        retrieved_documents: Optional[List[Document]],
        model_max_tokens: int,
        response_buffer: int = 4096
    ) -> str:
        """
        Builds the final prompt string from all components, ensuring it does not exceed the model's token limit.
        """

        prompt = ""

        # 1. System Prompt
        prompt += f"/no_think {system_prompt}\n\n"
        logger.info(f"Token count after adding system prompt: {count_tokens(prompt, self.is_ollama)}")

        # 2. Query
        prompt += f"\n\n<query>\n{query}\n</query>\n\nAssistant: "
        logger.info(f"Token count after adding query: {count_tokens(prompt, self.is_ollama)}")

        # 3. File Content
        if file_content:
            prompt += f"<currentFileContent path=\"{file_path}\">\n{file_content}\n</currentFileContent>\n\n"
            logger.info(f"Token count after adding file content: {count_tokens(prompt, self.is_ollama)}")

        # 4. Conversation History
        if conversation_history:
            prompt += f"<conversation_history>\n{conversation_history}\n</conversation_history>\n\n"
            logger.info(f"Token count after adding conversation history: {count_tokens(prompt, self.is_ollama)}")

        # 5. RAG documents
        if retrieved_documents:
            rag_content = "\n\n" + "-" * 10 + "\n\n".join([f"## File Path: {doc.meta_data.get('file_path', 'unknown')}\n\n{doc.text}" for doc in retrieved_documents])
            prompt += f"<START_OF_CONTEXT>\n{rag_content}\n<END_OF_CONTEXT>\n\n"
            logger.info(f"Token count after adding RAG documents: {count_tokens(prompt, self.is_ollama)}")

        # Final truncation
        final_token_count = count_tokens(prompt, self.is_ollama)
        if final_token_count > model_max_tokens - response_buffer:
            logger.warning(f"Final prompt token count ({final_token_count}) exceeds budget. Truncating...")
            prompt = self._truncate_text(prompt, model_max_tokens - response_buffer)
            final_token_count = count_tokens(prompt, self.is_ollama)
            logger.info(f"Truncated final prompt token count: {final_token_count}")

        return prompt
