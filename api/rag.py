import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import adalflow as adal
from adalflow.components.retriever.faiss_retriever import FAISSRetriever

from .config import configs
from .data_pipeline import DatabaseManager
from .tools.embedder import get_embedder

logger = logging.getLogger(__name__)

# Full, correct data classes
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
        self.dialog_turns = []
    def append_dialog_turn(self, dialog_turn):
        self.dialog_turns.append(dialog_turn)

class Memory(adal.core.component.DataComponent):
    def __init__(self):
        super().__init__()
        self.current_conversation = CustomConversation()
    def call(self) -> dict:
        return {turn.id: turn for turn in self.current_conversation.dialog_turns}
    def add_dialog_turn(self, user_query: str, assistant_response: str):
        turn = DialogTurn(id=str(len(self.current_conversation.dialog_turns)), user_query=UserQuery(query_str=user_query), assistant_response=AssistantResponse(response_str=assistant_response))
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
        self.transformed_docs = []
        self.retriever = None

    def _validate_and_filter_embeddings(self, documents: List) -> List:
        if not documents: return []
        valid_documents = []
        target_size = None
        for doc in documents:
            if hasattr(doc, 'vector') and doc.vector is not None:
                size = len(doc.vector)
                if target_size is None: target_size = size
                if size == target_size: valid_documents.append(doc)
        return valid_documents

    def prepare_retriever(self, repo_url_or_path: str, type: str = "github", access_token: str = None,
                      excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
        self.transformed_docs = self.db_manager.prepare_database(
            repo_url_or_path, type, access_token, self.is_ollama_embedder,
            excluded_dirs, excluded_files, included_dirs, included_files
        )
        self.transformed_docs = self._validate_and_filter_embeddings(self.transformed_docs)
        if not self.transformed_docs:
            raise ValueError("No valid documents with embeddings found after validation.")
        self.retriever = FAISSRetriever(
            **configs["retriever"],
            embedder=self.embedder,
            documents=self.transformed_docs,
            document_map_func=lambda doc: doc.vector,
        )

    def call(self, query: str, language: str = "en") -> Tuple[List, List]:
        try:
            if not self.retriever:
                raise RuntimeError("Retriever is not prepared.")
            retrieved_documents = self.retriever(query)
            retrieved_documents[0].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retrieved_documents[0].doc_indices
            ]
            return retrieved_documents, []
        except Exception as e:
            logger.error(f"Error in RAG call: {str(e)}")
            return ([], [])
