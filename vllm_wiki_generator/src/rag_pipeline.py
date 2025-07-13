import os
from .vllm_client import VLLMClient
from .file_processor import find_code_files, read_file_content
from .text_chunker import chunk_text
from langchain.vectorstores import FAISS
from langchain.embeddings import JinaEmbeddings

class RAGPipeline:
    """
    A RAG pipeline for answering questions about a codebase.
    """

    def __init__(self):
        """
        Initializes the RAGPipeline.
        """
        self.vllm_client = VLLMClient(
            model=os.getenv("VLLM_MODEL", "facebook/opt-125m"),
            base_url=os.getenv("VLLM_API_BASE"),
            api_key=os.getenv("VLLM_API_KEY")
        )
        self.embeddings = JinaEmbeddings(
            jina_api_key=os.getenv("JINA_API_KEY"),
            model_name=os.getenv("EMBEDDING_MODEL", "jina-embeddings-v2-base-en")
        )
        self.vector_store = None

    def _create_vector_store(self, repo_path: str):
        """
        Creates a vector store for a given repository.

        Args:
            repo_path (str): The path to the repository.
        """
        code_files = find_code_files(repo_path)
        chunks = []
        for file_path in code_files:
            file_content = read_file_content(file_path)
            file_chunks = chunk_text(file_content)
            chunks.extend(file_chunks)

        self.vector_store = FAISS.from_texts(chunks, self.embeddings)

    def ask(self, repo_path: str, query: str) -> str:
        """
        Answers a question about a repository.

        Args:
            repo_path (str): The path to the repository.
            query (str): The question to ask.

        Returns:
            str: The answer to the question.
        """
        if self.vector_store is None:
            self._create_vector_store(repo_path)

        docs = self.vector_store.similarity_search(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        answer = self.vllm_client.generate([prompt])[0]
        return answer
