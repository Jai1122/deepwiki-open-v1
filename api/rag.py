import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import adalflow as adal
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.core.types import Document

from .config import configs
from .data_pipeline import DatabaseManager
from .tools.embedder import get_embedder

logger = logging.getLogger(__name__)

@dataclass
class UserQuery:
    query_str: str

@dataclass
class AssistantResponse:
    response_str: str

@dataclass
class DialogTurn:
    id: str
    user_query: UserQuery
    assistant_response: AssistantResponse

class CustomConversation:
    def __init__(self):
        self.dialog_turns: List[DialogTurn] = []

    def append_dialog_turn(self, dialog_turn: DialogTurn):
        self.dialog_turns.append(dialog_turn)

class Memory(adal.core.component.DataComponent):
    def __init__(self):
        super().__init__()
        self.current_conversation = CustomConversation()

    def call(self) -> dict:
        return {turn.id: turn for turn in self.current_conversation.dialog_turns}

    def add_dialog_turn(self, user_query: str, assistant_response: str):
        turn_id = str(len(self.current_conversation.dialog_turns))
        turn = DialogTurn(
            id=turn_id,
            user_query=UserQuery(query_str=user_query),
            assistant_response=AssistantResponse(response_str=assistant_response)
        )
        self.current_conversation.append_dialog_turn(turn)

@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str = field(default="")
    answer: str = field(default="")
    __output_fields__ = ["rationale", "answer"]

class RAG(adal.Component):
    def __init__(self, provider="google", model=None, use_s3: bool = False):
        super().__init__()
        self.provider = provider
        self.model = model
        from .config import is_ollama_embedder
        self.is_ollama_embedder = is_ollama_embedder()
        self.memory = Memory()
        self.embedder = get_embedder()
        self.db_manager = DatabaseManager()
        self.transformed_docs: List[Document] = []
        self.retriever: Optional[FAISSRetriever] = None

    def _validate_and_filter_embeddings(self, documents: List[Document]) -> List[Document]:
        """
        Ensures all documents have embeddings of a consistent dimension.
        """
        if not documents:
            return []
            
        valid_documents = []
        target_size = None
        
        # First pass to determine the target embedding size from the first valid doc
        for doc in documents:
            if hasattr(doc, 'vector') and doc.vector is not None:
                target_size = len(doc.vector)
                break
        
        if target_size is None:
            logger.warning("No documents with valid embeddings found.")
            return []

        # Second pass to filter documents based on the target size
        for doc in documents:
            if hasattr(doc, 'vector') and doc.vector is not None and len(doc.vector) == target_size:
                valid_documents.append(doc)
            else:
                logger.warning(f"Skipping document with inconsistent or missing embedding. Source: {doc.metadata.get('source', 'N/A')}")

        return valid_documents

    def prepare_retriever(self, repo_url_or_path: str, type: str = "github", access_token: str = None,
                      excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
        """
        Prepares the FAISS retriever by processing the repository and building the index.
        """
        try:
            self.transformed_docs = self.db_manager.prepare_database(
                repo_url_or_path, type, access_token, self.is_ollama_embedder,
                excluded_dirs, excluded_files, included_dirs, included_files
            )
        except Exception as e:
            logger.error(f"Failed to prepare database: {e}", exc_info=True)
            self.transformed_docs = [] # Ensure it's a list on failure

        if not self.transformed_docs:
            logger.warning("prepare_database returned no documents. Retriever will not be available.")
            self.transformed_docs = []
            return # Exit early if no docs

        self.transformed_docs = self._validate_and_filter_embeddings(self.transformed_docs)
        
        if not self.transformed_docs:
            logger.error("No valid documents with embeddings found after validation. Cannot build retriever.")
            raise ValueError("No valid documents with embeddings found after validation.")
            
        retriever_config = configs.get("retriever")
        if not isinstance(retriever_config, dict):
            logger.error("Retriever configuration is missing or invalid in embedder.json.")
            raise ValueError("Retriever configuration is missing or invalid.")

        logger.info(f"Initializing FAISS retriever with {len(self.transformed_docs)} documents.")
        self.retriever = FAISSRetriever(
            **retriever_config,
            embedder=self.embedder,
            documents=self.transformed_docs,
            document_map_func=lambda doc: doc.vector,
        )

    def call(self, query: str, language: str = "en") -> Tuple[List, List]:
        """
        Performs a retrieval call.
        Returns a tuple containing the list of retrieved documents and an empty list.
        """
        if not self.retriever:
            logger.warning("Retriever is not prepared. Returning empty result.")
            return ([], [])
            
        try:
            # The retriever call returns a list of documents
            retrieved_results = self.retriever(query)
            
            if not retrieved_results:
                logger.info("Retriever returned no results for the query.")
                return ([], [])

            # The actual documents are in the 'documents' attribute of the first result item
            # This assumes the retriever returns a list with one main result object
            retrieved_documents = retrieved_results[0].documents
            
            return retrieved_documents, []
        except Exception as e:
            logger.error(f"Error in RAG call: {e}", exc_info=True)
            return ([], [])
