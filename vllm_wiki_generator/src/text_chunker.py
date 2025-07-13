import tiktoken
from typing import List

def chunk_text(text: str, chunk_size: int = 2048, overlap: int = 256) -> List[str]:
    """
    Splits a text into overlapping chunks based on token count.

    Args:
        text (str): The text to split.
        chunk_size (int, optional): The desired size of each chunk in tokens. Defaults to 2048.
        overlap (int, optional): The desired overlap between chunks in tokens. Defaults to 256.

    Returns:
        List[str]: A list of text chunks.
    """
    if chunk_size <= overlap:
        raise ValueError("Chunk size must be greater than overlap.")

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end >= len(tokens):
            break

        start += chunk_size - overlap

    return chunks
