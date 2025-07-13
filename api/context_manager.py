import logging
from typing import List
from adalflow.core.types import Document
from api.data_pipeline import count_tokens

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages the context provided to the LLM by intelligently selecting,
    truncating, and formatting documents to fit within a specified token limit.
    """

    def __init__(self, model_provider: str):
        """
        Initializes the ContextManager.

        Args:
            model_provider (str): The name of the model provider (e.g., 'ollama', 'openai'),
                                  which can affect token counting.
        """
        self.is_ollama = model_provider == "ollama"

    def build_context(self, documents: List[Document], max_tokens: int) -> str:
        """
        Builds a context string from a list of documents, ensuring it does not exceed max_tokens.
        It uses a "stuffing" and truncation strategy.

        Args:
            documents (List[Document]): A list of documents, assumed to be sorted by relevance.
            max_tokens (int): The maximum number of tokens the final context string can have.

        Returns:
            str: A formatted context string ready to be inserted into a prompt.
        """
        if not documents:
            logger.warning("No documents provided to build context.")
            return ""

        context_parts = []
        current_tokens = 0

        # Header for the context block
        context_header = "<START_OF_CONTEXT>\n"
        context_footer = "\n<END_OF_CONTEXT>"
        # Reserve tokens for headers, footers, and separators
        reserved_tokens = count_tokens(context_header + context_footer, self.is_ollama)

        token_budget = max_tokens - reserved_tokens

        logger.info(f"Building context with a budget of {token_budget} tokens.")

        for doc in documents:
            file_path = doc.meta_data.get('file_path', 'unknown')
            doc_header = f"## File Path: {file_path}\n\n"

            # Calculate tokens for the document's header and content
            header_tokens = count_tokens(doc_header, self.is_ollama)
            content_tokens = count_tokens(doc.text, self.is_ollama)
            total_doc_tokens = header_tokens + content_tokens

            # Check if the entire document fits within the remaining budget
            if current_tokens + total_doc_tokens <= token_budget:
                context_parts.append(f"{doc_header}{doc.text}")
                current_tokens += total_doc_tokens + count_tokens("\n\n" + "-" * 10 + "\n\n", self.is_ollama) # Separator tokens
            else:
                # If the doc doesn't fit, see if we can truncate it
                remaining_budget = token_budget - current_tokens
                # Only add the document if there's enough space for its header and some content
                if remaining_budget > header_tokens + 50: # Require at least 50 tokens for content

                    available_content_tokens = remaining_budget - header_tokens

                    # Simple character-based truncation assuming ~4 chars/token as a rough guide
                    # A more precise method would be to encode and slice the tokens, but this is a robust approximation.
                    avg_chars_per_token = 4
                    estimated_chars_to_keep = int(available_content_tokens * avg_chars_per_token)

                    truncated_content = doc.text[:estimated_chars_to_keep]

                    # Re-count tokens to be precise after truncation
                    final_truncated_content = truncated_content
                    while count_tokens(final_truncated_content, self.is_ollama) > available_content_tokens:
                        # Reduce by 10% until it fits
                        final_truncated_content = final_truncated_content[:int(len(final_truncated_content) * 0.9)]

                    context_parts.append(f"{doc_header}{final_truncated_content}...")
                    logger.warning(f"Document '{file_path}' was truncated to fit the context window.")
                    current_tokens += count_tokens(context_parts[-1], self.is_ollama)

                # Stop processing more documents as we have hit the token limit
                logger.info("Context budget reached. Stopping document processing.")
                break

        if not context_parts:
            logger.warning("No documents could fit into the specified token budget.")
            return ""

        # Join all parts with clear separation
        final_context = "\n\n" + "-" * 10 + "\n\n".join(context_parts)

        # Final check
        final_tokens = count_tokens(context_header + final_context + context_footer, self.is_ollama)
        logger.info(f"Final context built with {len(context_parts)} documents and {final_tokens} tokens (limit: {max_tokens}).")

        return context_header + final_context + context_footer
