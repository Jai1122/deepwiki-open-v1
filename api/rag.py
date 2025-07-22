import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Dict
from uuid import uuid4

import adalflow as adal

from .tools.embedder import get_embedder
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from .config import configs
from .data_pipeline import DatabaseManager

# Configure logging
logger = logging.getLogger(__name__)

# Corrected Conversation classes
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
        if not hasattr(self, 'dialog_turns'):
            self.dialog_turns = []
        self.dialog_turns.append(dialog_turn)

class Memory(adal.core.component.DataComponent):
    def __init__(self):
        super().__init__()
        self.current_conversation = CustomConversation()

    def call(self) -> Dict:
        all_dialog_turns = {}
        if hasattr(self.current_conversation, 'dialog_turns') and self.current_conversation.dialog_turns:
            for turn in self.current_conversation.dialog_turns:
                if hasattr(turn, 'id') and turn.id is not None:
                    all_dialog_turns[turn.id] = turn
        return all_dialog_turns

    def add_dialog_turn(self, user_query: str, assistant_response: str) -> bool:
        try:
            dialog_turn = DialogTurn(
                id=str(uuid4()),
                user_query=UserQuery(query_str=user_query),
                assistant_response=AssistantResponse(response_str=assistant_response),
            )
            self.current_conversation.dialog_turns.append(dialog_turn)
            return True
        except Exception as e:
            logger.error(f"Error adding dialog turn: {str(e)}")
            return False

@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str = field(default="", metadata={"desc": "Chain of thoughts for the answer."})
    answer: str = field(default="", metadata={"desc": "Answer to the user query."})
    __output_fields__ = ["rationale", "answer"]

class RAG(adal.Component):
    def __init__(self, provider="google", model=None, use_s3: bool = False):
        super().__init__()
        self.provider = provider
        self.model = model
        from .config import get_embedder_config, is_ollama_embedder
        self.is_ollama_embedder = is_ollama_embedder()
        if self.is_ollama_embedder:
            from .ollama_patch import check_ollama_model_exists
            embedder_config = get_embedder_config()
            if embedder_config and embedder_config.get("model_kwargs", {}).get("model"):
                model_name = embedder_config["model_kwargs"]["model"]
                if not check_ollama_model_exists(model_name):
                    raise Exception(f"Ollama model '{model_name}' not found.")
        self.memory = Memory()
        self.embedder = get_embedder()
        def single_string_embedder(query):
            if isinstance(query, list):
                query = query[0]
            return self.embedder(input=query)
        self.query_embedder = single_string_embedder if self.is_ollama_embedder else self.embedder
        self.initialize_db_manager()
        data_parser = adal.DataClassParser(data_class=RAGAnswer, return_data_class=True)
        from .config import get_model_config
        generator_config = get_model_config(self.provider, self.model)
        init_kwargs = generator_config.get("initialize_kwargs", {})
        self.generator = adal.Generator(
            template="", # Placeholder
            prompt_kwargs={},
            model_client=generator_config["model_client"](**init_kwargs),
            model_kwargs=generator_config["model_kwargs"],
            output_processors=data_parser,
        )

    def initialize_db_manager(self):
        self.db_manager = DatabaseManager()
        self.transformed_docs = []

    def _validate_and_filter_embeddings(self, documents: List) -> List:
        if not documents:
            return []
        valid_documents = []
        target_size = None
        for i, doc in enumerate(documents):
            if hasattr(doc, 'vector') and doc.vector is not None:
                size = len(doc.vector)
                if target_size is None:
                    target_size = size
                if size == target_size:
                    valid_documents.append(doc)
                else:
                    logger.warning(f"Skipping document with mismatched embedding size. Expected {target_size}, got {size}.")
            else:
                logger.warning(f"Skipping document with no embedding vector.")
        return valid_documents

    def prepare_retriever(self, repo_url_or_path: str, type: str = "github", access_token: str = None,
                      excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
        self.initialize_db_manager()
        self.repo_url_or_path = repo_url_or_path
        self.transformed_docs = self.db_manager.prepare_database(
            repo_url_or_path, type, access_token, self.is_ollama_embedder,
            excluded_dirs, excluded_files, included_dirs, included_files
        )
        
        # --- CRITICAL FIX: RESTORED VALIDATION CALL ---
        self.transformed_docs = self._validate_and_filter_embeddings(self.transformed_docs)

        if not self.transformed_docs:
            raise ValueError("No valid documents with embeddings found after validation.")
        
        retrieve_embedder = self.query_embedder if self.is_ollama_embedder else self.embedder
        self.retriever = FAISSRetriever(
            **configs["retriever"],
            embedder=retrieve_embedder,
            documents=self.transformed_docs,
            document_map_func=lambda doc: doc.vector,
        )

    def call(self, query: str, language: str = "en") -> Tuple[List, List]:
        try:
            retrieved_documents = self.retriever(query)
            retrieved_documents[0].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retrieved_documents[0].doc_indices
            ]
            return retrieved_documents, []
        except Exception as e:
            logger.error(f"Error in RAG call: {str(e)}")
            error_answer = RAGAnswer(
                rationale="Error during retrieval.",
                answer="I apologize, but I encountered an error while retrieving relevant documents."
            )
            return [error_answer], []