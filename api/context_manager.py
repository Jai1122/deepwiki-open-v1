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

    def _chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """A helper to chunk text into parts of a max token count."""
        if not text or max_tokens <= 0:
            return []

        encoding = get_tokenizer(self.is_ollama)
        tokens = encoding.encode(text)

        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunks.append(encoding.decode(chunk_tokens))

        return chunks

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
    ) -> List[str]:
        """
        Builds the final prompt string from all components, ensuring it does not exceed the model's token limit.
        This implementation uses a proactive token budgeting approach and chunks the file content if necessary.
        """
        # Set the total budget for the prompt
        total_budget = model_max_tokens - response_buffer
        if total_budget <= 0:
            logger.error("Error in context_manager.py: Model max tokens is less than or equal to the response buffer. No budget for prompt.")
            return []

        # 1. System Prompt and Query (Static components, must be included)
        # These are considered essential and are not truncated.
        # The initial prompt includes the system prompt, query, and placeholders for other components.
        prompt_template = "/no_think {system_prompt}\n\n<query>\n{query}\n</query>\n\nAssistant: {file_content_str}{conversation_history_str}{rag_content_str}"

        # Calculate the base token count for the static components
        base_prompt = prompt_template.format(
            system_prompt=system_prompt,
            query=query,
            file_content_str="",
            conversation_history_str="",
            rag_content_str=""
        )
        base_tokens = count_tokens(base_prompt, self.is_ollama)

        # The remaining budget is for dynamic components (file content, conversation history, RAG documents)
        remaining_budget = total_budget - base_tokens
        logger.info(f"Initial budget: {total_budget} tokens. Remaining for dynamic content: {remaining_budget} tokens.")

        # 2. File Content (High priority)
        file_content_chunks = []
        if file_content and remaining_budget > 0:
            template = f"<currentFileContent path=\"{file_path}\">\n{{content}}\n</currentFileContent>\n\n"
            content_placeholder = "{content}"

            # Calculate the token count for the template without the actual file content
            template_tokens = count_tokens(template.replace(content_placeholder, ""), self.is_ollama)

            # The budget for the file content is the remaining budget minus the template tokens
            file_content_budget = remaining_budget - template_tokens

            # Chunk file content if it exceeds the budget
            file_content_chunks = self._chunk_text(file_content, file_content_budget)

        # 3. Conversation History (Medium priority)
        conversation_history_str = ""
        if conversation_history and remaining_budget > 0:
            template = f"<conversation_history>\n{{content}}\n</conversation_history>\n\n"
            content_placeholder = "{content}"

            template_tokens = count_tokens(template.replace(content_placeholder, ""), self.is_ollama)
            history_budget = remaining_budget - template_tokens

            truncated_history = self._truncate_text(conversation_history, history_budget)

            conversation_history_str = template.format(content=truncated_history)
            remaining_budget -= count_tokens(conversation_history_str, self.is_ollama)
            logger.info(f"Token count after adding conversation history: {count_tokens(conversation_history_str, self.is_ollama)}. Remaining budget: {remaining_budget} tokens.")

        # 4. RAG documents (Low priority)
        rag_content_str = ""
        if retrieved_documents and remaining_budget > 0:
            template = "<START_OF_CONTEXT>\n{content}\n<END_OF_CONTEXT>\n\n"
            content_placeholder = "{content}"

            template_tokens = count_tokens(template.replace(content_placeholder, ""), self.is_ollama)
            rag_budget = remaining_budget - template_tokens

            rag_content = "\n\n" + "-" * 10 + "\n\n".join([f"## File Path: {doc.meta_data.get('file_path', 'unknown')}\n\n{doc.text}" for doc in retrieved_documents])
            truncated_rag_content = self._truncate_text(rag_content, rag_budget)

            rag_content_str = template.format(content=truncated_rag_content)
            remaining_budget -= count_tokens(rag_content_str, self.is_ollama)
            logger.info(f"Token count after adding RAG documents: {count_tokens(rag_content_str, self.is_ollama)}. Remaining budget: {remaining_budget} tokens.")

        prompts = []
        if not file_content_chunks:
            # If there are no file content chunks, create a single prompt
            file_content_chunks.append("")

        for chunk in file_content_chunks:
            file_content_str = ""
            if chunk:
                template = f"<currentFileContent path=\"{file_path}\">\n{{content}}\n</currentFileContent>\n\n"
                file_content_str = template.format(content=chunk)

            # Assemble the final prompt
            final_prompt = prompt_template.format(
                system_prompt=system_prompt,
                query=query,
                file_content_str=file_content_str,
                conversation_history_str=conversation_history_str,
                rag_content_str=rag_content_str
            )

            # Final check to ensure the prompt does not exceed the budget
            final_token_count = count_tokens(final_prompt, self.is_ollama)
            if final_token_count > total_budget:
                logger.critical(f"Error in context_manager.py - CRITICAL: Final prompt token count ({final_token_count}) exceeded budget after assembly. This indicates a bug in context manager")
                # Fallback to truncating the entire prompt to prevent an error from being sent to the model
                final_prompt = self._truncate_text(final_prompt, total_budget)

            prompts.append(final_prompt)

        logger.info(f"Generated {len(prompts)} prompts.")
        return prompts
