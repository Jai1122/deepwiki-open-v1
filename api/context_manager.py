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

        # Use a more precise token-based truncation
        from api.data_pipeline import get_tokenizer
        encoding = get_tokenizer(self.is_ollama)
        tokens = encoding.encode(text)

        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            truncated_text = encoding.decode(truncated_tokens)
        else:
            truncated_text = text

        # Re-count tokens to be precise after truncation and adjust if necessary
        while count_tokens(truncated_text, self.is_ollama) > max_tokens:
            # Reduce by 20% until it fits
            truncated_text = truncated_text[:int(len(truncated_text) * 0.8)]

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
        response_buffer: int = 4096
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
        logger.info(f"System prompt tokens: {count_tokens(system_prompt, self.is_ollama)}")
        logger.info(f"Query tokens: {count_tokens(query, self.is_ollama)}")

        # --- 3. Dynamic Truncation ---
        components = []
        if file_content:
            components.append({"name": "file_content", "content": file_content, "priority": 1})
        if conversation_history:
            components.append({"name": "conversation_history", "content": conversation_history, "priority": 2})
        if retrieved_documents:
            rag_content = rag_doc_separator.join([f"## File Path: {doc.meta_data.get('file_path', 'unknown')}\n\n{doc.text}" for doc in retrieved_documents])
            components.append({"name": "rag_documents", "content": rag_content, "priority": 3})

        total_content_tokens = sum(count_tokens(c["content"], self.is_ollama) for c in components)

        if total_content_tokens > available_budget:
            # Distribute the budget according to priority
            for component in sorted(components, key=lambda x: x["priority"]):
                if available_budget <= 0:
                    component["content"] = ""
                    continue

                component_tokens = count_tokens(component["content"], self.is_ollama)
                if component_tokens > available_budget:
                    component["content"] = self._truncate_text(component["content"], available_budget)
                    available_budget = 0
                else:
                    available_budget -= component_tokens

        final_components = []
        for component in components:
            if component["name"] == "file_content" and component["content"]:
                final_components.append(file_content_wrapper.format(file_path, component["content"]))
            elif component["name"] == "conversation_history" and component["content"]:
                final_components.append(history_wrapper.format(component["content"]))
            elif component["name"] == "rag_documents" and component["content"]:
                final_components.append(rag_context_wrapper.format(component["content"]))

        # --- 4. Assemble the final prompt ---

        prompt = prompt_template_start
        prompt += "".join(final_components)
        if not retrieved_documents and not file_content:
             prompt += "<note>Answering without retrieval augmentation or file content.</note>\n\n"
        prompt += prompt_template_end

        final_token_count = count_tokens(prompt, self.is_ollama)
        logger.info(f"Final prompt assembled with {final_token_count} tokens (Model Max: {model_max_tokens}, Budget: {model_max_tokens - response_buffer}).")
        logger.info(f"Final prompt: {prompt}")

        if final_token_count > (model_max_tokens - response_buffer):
            logger.error(f"CRITICAL: Final prompt token count ({final_token_count}) exceeded budget after assembly. This indicates a bug in the ContextManager.")
            # Fallback to a safe, minimal prompt
            return self._truncate_text(prompt, model_max_tokens - response_buffer)

        return prompt
