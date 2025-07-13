import logging
from typing import List, Optional
from adalflow.core.types import Document
from api.data_pipeline import count_tokens

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

        # Simple character-based truncation assuming ~4 chars/token as a rough guide
        # This is a robust approximation. A more precise method would be to encode, slice tokens, and decode.
        avg_chars_per_token = 4
        estimated_chars_to_keep = int(max_tokens * avg_chars_per_token)

        truncated_text = text[:estimated_chars_to_keep]

        # Re-count tokens to be precise after truncation and adjust if necessary
        while count_tokens(truncated_text, self.is_ollama) > max_tokens:
            # Reduce by 10% until it fits
            truncated_text = truncated_text[:int(len(truncated_text) * 0.9)]

        return truncated_text

    def build_prompt(
        self,
        system_prompt: str,
        query: str,
        conversation_history: str,
        file_content: Optional[str],
        file_path: Optional[str],
        retrieved_documents: Optional[List[Document]],
        model_max_tokens: int,
        response_buffer: int = 2048
    ) -> str:
        """
        Builds the final prompt string from all components, ensuring it does not exceed the model's token limit.
        It uses a prioritized stuffing and truncation strategy.

        Args:
            system_prompt (str): The system prompt.
            query (str): The user's query.
            conversation_history (str): The formatted string of previous turns.
            file_content (Optional[str]): The content of the user-specified file.
            file_path (Optional[str]): The path of the user-specified file.
            retrieved_documents (Optional[List[Document]]): Documents from RAG, sorted by relevance.
            model_max_tokens (int): The absolute maximum token limit of the target model.
            response_buffer (int): A buffer of tokens to leave for the model's response.

        Returns:
            str: A formatted and size-guaranteed prompt string.
        """
        # --- 1. Define all template parts and their token costs ---

        # Base template parts that are always present
        prompt_template_start = f"/no_think {system_prompt}\n\n"
        prompt_template_end = f"\n\n<query>\n{query}\n</query>\n\nAssistant: "

        # Optional component wrappers
        history_wrapper = "<conversation_history>\n{}\n</conversation_history>\n\n"
        file_content_wrapper = "<currentFileContent path=\"{}\">\n{}\n</currentFileContent>\n\n"
        rag_context_wrapper = "<START_OF_CONTEXT>\n{}\n<END_OF_CONTEXT>\n\n"
        rag_doc_separator = "\n\n" + "-" * 10 + "\n\n"

        # Calculate the token cost of the static parts of the template
        static_template_tokens = count_tokens(prompt_template_start + prompt_template_end, self.is_ollama)

        # --- 2. Calculate the available budget for dynamic content ---

        available_budget = model_max_tokens - static_template_tokens - response_buffer
        logger.info(f"Initial budget for dynamic content: {available_budget} tokens.")

        # --- 3. Prioritized Stuffing and Truncation ---

        final_components = []

        # Priority 1: File Content (if provided)
        if file_content:
            wrapper_cost = count_tokens(file_content_wrapper.format(file_path, ""), self.is_ollama)
            content_budget = available_budget - wrapper_cost

            if content_budget > 0:
                truncated_file_content = self._truncate_text(file_content, content_budget)
                if truncated_file_content:
                    final_file_component = file_content_wrapper.format(file_path, truncated_file_content)
                    final_components.append(final_file_component)
                    available_budget -= count_tokens(final_file_component, self.is_ollama)
                    if len(truncated_file_content) < len(file_content):
                        logger.warning(f"File content for '{file_path}' was truncated to fit the context window.")
                else:
                    logger.warning(f"Not enough budget to include any content from file '{file_path}'.")
            else:
                logger.warning(f"Not enough budget to include file content wrapper for '{file_path}'.")

        # Priority 2: Conversation History
        if conversation_history:
            wrapper_cost = count_tokens(history_wrapper.format(""), self.is_ollama)
            content_budget = available_budget - wrapper_cost

            if content_budget > 0:
                truncated_history = self._truncate_text(conversation_history, content_budget)
                if truncated_history:
                    final_history_component = history_wrapper.format(truncated_history)
                    final_components.append(final_history_component)
                    available_budget -= count_tokens(final_history_component, self.is_ollama)
                    if len(truncated_history) < len(conversation_history):
                        logger.warning("Conversation history was truncated to fit the context window.")
                else:
                    logger.warning("Not enough budget to include any conversation history.")
            else:
                logger.warning("Not enough budget to include conversation history wrapper.")

        # Priority 3: RAG Context (retrieved documents)
        rag_content_parts = []
        if retrieved_documents:
            wrapper_cost = count_tokens(rag_context_wrapper.format(""), self.is_ollama)
            content_budget = available_budget - wrapper_cost

            if content_budget > 0:
                rag_token_counter = 0
                for doc in retrieved_documents:
                    doc_path = doc.meta_data.get('file_path', 'unknown')
                    doc_header = f"## File Path: {doc_path}\n\n"
                    doc_separator_cost = count_tokens(rag_doc_separator, self.is_ollama)

                    doc_cost = count_tokens(doc_header + doc.text, self.is_ollama) + doc_separator_cost

                    if rag_token_counter + doc_cost <= content_budget:
                        rag_content_parts.append(f"{doc_header}{doc.text}")
                        rag_token_counter += doc_cost
                    else:
                        # Truncate the last document that partially fits
                        remaining_doc_budget = content_budget - rag_token_counter - count_tokens(doc_header, self.is_ollama) - doc_separator_cost
                        if remaining_doc_budget > 50: # Minimum meaningful content size
                            truncated_doc_text = self._truncate_text(doc.text, remaining_doc_budget)
                            rag_content_parts.append(f"{doc_header}{truncated_doc_text}...")
                            logger.warning(f"RAG document '{doc_path}' was truncated.")
                        break # Budget is full

                if rag_content_parts:
                    final_rag_content = rag_doc_separator.join(rag_content_parts)
                    final_rag_component = rag_context_wrapper.format(final_rag_content)
                    final_components.append(final_rag_component)
                    available_budget -= count_tokens(final_rag_component, self.is_ollama)
            else:
                logger.warning("Not enough budget to include RAG context wrapper.")

        # --- 4. Assemble the final prompt ---

        prompt = prompt_template_start
        prompt += "".join(final_components)
        if not retrieved_documents and not file_content:
             prompt += "<note>Answering without retrieval augmentation or file content.</note>\n\n"
        prompt += prompt_template_end

        final_token_count = count_tokens(prompt, self.is_ollama)
        logger.info(f"Final prompt assembled with {final_token_count} tokens (Model Max: {model_max_tokens}, Budget: {model_max_tokens - response_buffer}).")

        if final_token_count > (model_max_tokens - response_buffer):
            logger.error(f"CRITICAL: Final prompt token count ({final_token_count}) exceeded budget after assembly. This indicates a bug in the ContextManager.")
            # Fallback to a safe, minimal prompt
            return f"{prompt_template_start}{prompt_template_end}"

        return prompt
```
