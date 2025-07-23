import logging
import tiktoken

logger = logging.getLogger(__name__)

def count_tokens(text: str, is_ollama_embedder: bool = False) -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens with tiktoken: {e}")
        return len(text) // 4

def truncate_prompt_to_fit(
    max_tokens: int,
    system_prompt: str,
    conversation_history: str,
    file_content: str,
    context_text: str,
    query: str,
    is_ollama: bool = False
) -> (str, str):
    """
    Intelligently truncates file_content and context_text to fit within the model's max_token limit.
    """
    fixed_components = [system_prompt, conversation_history, query]
    fixed_tokens = sum(count_tokens(text, is_ollama) for text in fixed_components)
    
    reserved_tokens = 2048
    available_tokens = max_tokens - fixed_tokens - reserved_tokens
    
    if available_tokens <= 0:
        logger.warning("Not enough tokens for context and file content after reserving space.")
        return "", ""

    file_tokens = count_tokens(file_content, is_ollama)
    context_tokens = count_tokens(context_text, is_ollama)
    total_variable_tokens = file_tokens + context_tokens

    if total_variable_tokens <= available_tokens:
        return file_content, context_text

    if file_content and context_text:
        file_alloc = int(available_tokens * 0.7)
        context_alloc = available_tokens - file_alloc
    elif file_content:
        file_alloc = available_tokens
        context_alloc = 0
    else:
        file_alloc = 0
        context_alloc = available_tokens

    truncated_file_content = file_content
    if count_tokens(truncated_file_content, is_ollama) > file_alloc:
        while count_tokens(truncated_file_content, is_ollama) > file_alloc:
            truncated_file_content = truncated_file_content[:int(len(truncated_file_content) * 0.9)]
    
    truncated_context_text = context_text
    if count_tokens(truncated_context_text, is_ollama) > context_alloc:
        while count_tokens(truncated_context_text, is_ollama) > context_alloc:
            truncated_context_text = truncated_context_text[:int(len(truncated_context_text) * 0.9)]

    return truncated_file_content, truncated_context_text
