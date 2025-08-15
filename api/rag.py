import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import adalflow as adal
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.core.types import Document

from .config import configs

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
    def __init__(self, provider="vllm", model=None, use_s3: bool = False):
        super().__init__()
        self.provider = provider
        self.model = model
        self.memory = Memory()
        self._embedder = None  # Lazy loading to avoid circular imports
        self._db_manager = None  # Lazy loading to avoid circular imports
        self.transformed_docs: List[Document] = []
        self.retriever: Optional[FAISSRetriever] = None

    @property
    def embedder(self):
        """Lazy loading of embedder to avoid circular import issues"""
        if self._embedder is None:
            from .tools.embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    @property
    def db_manager(self):
        """Lazy loading of database manager to avoid circular import issues"""
        if self._db_manager is None:
            from .data_pipeline import DatabaseManager
            self._db_manager = DatabaseManager()
        return self._db_manager

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

    def prepare_retriever(self, repo_url_or_path: str, type: str = "bitbucket", access_token: str = None):
        """
        Prepares the FAISS retriever by processing the repository and building the index.
        """
        try:
            # Enhanced processing for remote repositories to ensure comprehensive wiki generation
            if type == "bitbucket":
                # Use higher token limits and enhanced processing for remote repos
                logger.info(f"🚀 Enhanced RAG processing for {type} repository")
                self.transformed_docs = self.db_manager.prepare_database(
                    repo_url_or_path, type, access_token, max_total_tokens=3000000, prioritize_files=True
                )
            else:
                self.transformed_docs = self.db_manager.prepare_database(
                    repo_url_or_path, type, access_token
                )
        except Exception as e:
            logger.error(f"Failed to prepare database: {e}", exc_info=True)
            self.transformed_docs = [] # Ensure it's a list on failure
            raise RuntimeError(f"Database preparation failed: {e}")

        if not self.transformed_docs:
            error_msg = "prepare_database returned no documents. Retriever will not be available."
            logger.error(error_msg)
            self.transformed_docs = []
            raise RuntimeError(error_msg)

        self.transformed_docs = self._validate_and_filter_embeddings(self.transformed_docs)
        
        if not self.transformed_docs:
            logger.error("No valid documents with embeddings found after validation. Cannot build retriever.")
            raise ValueError("No valid documents with embeddings found after validation.")
            
        # Additional validation for document vectors
        vectors_ok = 0
        for i, doc in enumerate(self.transformed_docs[:5]):  # Check first 5 docs
            if hasattr(doc, 'vector') and doc.vector is not None:
                vectors_ok += 1
                logger.debug(f"Document {i}: vector dim={len(doc.vector)}")
            else:
                logger.warning(f"Document {i}: missing or None vector")
        
        logger.info(f"✅ Document validation: {vectors_ok}/{min(5, len(self.transformed_docs))} documents have valid vectors")
            
        retriever_config = configs.get("retriever")
        if not isinstance(retriever_config, dict):
            logger.error("Retriever configuration is missing or invalid in embedder.json.")
            raise ValueError("Retriever configuration is missing or invalid.")

        logger.info(f"Initializing FAISS retriever with {len(self.transformed_docs)} documents.")
        logger.debug(f"Retriever config: {retriever_config}")
        logger.debug(f"Embedder type: {self.embedder.__class__.__name__}")
        logger.debug(f"Sample document vector shape: {len(self.transformed_docs[0].vector) if self.transformed_docs and hasattr(self.transformed_docs[0], 'vector') and self.transformed_docs[0].vector else 'No vector'}")
        
        try:
            # Test document_map_func before passing to FAISS
            test_vectors = []
            for i, doc in enumerate(self.transformed_docs[:3]):  # Test first 3 docs
                try:
                    vector = doc.vector
                    if vector is not None:
                        test_vectors.append(len(vector))
                        logger.debug(f"Doc {i}: vector length {len(vector)}")
                    else:
                        logger.warning(f"Doc {i}: vector is None")
                except Exception as e:
                    logger.warning(f"Doc {i}: error accessing vector: {e}")
            
            logger.info(f"🧪 Vector validation: {len(test_vectors)} valid vectors out of {min(3, len(self.transformed_docs))} docs")
            
            self.retriever = FAISSRetriever(
                **retriever_config,
                embedder=self.embedder,
                documents=self.transformed_docs,
                document_map_func=lambda doc: doc.vector,
            )
            logger.info(f"✅ FAISS retriever initialized successfully with {len(self.transformed_docs)} documents")
        except Exception as e:
            logger.error(f"❌ FAISS retriever initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize FAISS retriever: {e}")

    def call(self, query: str, language: str = None) -> Tuple[List, List]:
        """
        Performs a retrieval call.
        
        Args:
            query (str): The search query
            language (str, optional): Language parameter (currently not used but accepted for compatibility)
        
        Returns:
            tuple: A tuple containing the list of retrieved documents and an empty list.
        """
        if not self.retriever:
            logger.error("❌ RAG CRITICAL ERROR: Retriever is not prepared. Cannot perform retrieval.")
            raise RuntimeError("RAG retriever not initialized. Call prepare_retriever() first.")
            
        try:
            # To ensure the query vector is generated in the exact same way as the
            # document vectors, we will wrap the query in a Document object and
            # pass it through the same ToEmbeddings pipeline component.
            from adalflow.components.data_process import ToEmbeddings
            from adalflow.core.types import Document

            # 1. Create a single-document pipeline
            query_pipeline = ToEmbeddings(embedder=self.embedder)
            
            # 2. Validate query length to prevent token limit errors
            from .utils import count_tokens
            query_tokens = count_tokens(query)
            max_query_tokens = 4000  # Conservative limit for queries
            
            if query_tokens > max_query_tokens:
                logger.warning(f"Query is too long ({query_tokens} tokens), truncating to {max_query_tokens} tokens")
                # Truncate query to fit within limits
                ratio = max_query_tokens / query_tokens
                query = query[:int(len(query) * ratio * 0.9)]  # 0.9 for safety margin
                logger.info(f"Truncated query to {count_tokens(query)} tokens")
            
            # 3. Wrap the query string in a Document object
            query_document = Document(text=query)
            
            # 4. Process the document to get the embedding
            transformed_query_doc = query_pipeline([query_document])
            
            if not transformed_query_doc or not hasattr(transformed_query_doc[0], 'vector'):
                raise ValueError("Query embedding pipeline did not return a valid vector.")

            query_vector = transformed_query_doc[0].vector
            
            # Validate query vector dimensions
            query_dim = len(query_vector) if query_vector else 0
            logger.debug(f"Query vector dimensions: {query_dim}")
            
            # Get expected dimensions from the first stored document
            expected_dim = None
            if self.transformed_docs:
                for doc in self.transformed_docs:
                    if hasattr(doc, 'vector') and doc.vector:
                        expected_dim = len(doc.vector)
                        break
            
            logger.debug(f"Expected vector dimensions: {expected_dim}")
            
            # Check for dimension mismatch
            if expected_dim and query_dim != expected_dim:
                logger.error(f"Dimension mismatch: Query={query_dim}, Expected={expected_dim}")
                logger.error("This usually means:")
                logger.error("  - Different embedding models were used for documents vs queries")
                logger.error("  - EMBEDDING_DIMENSIONS setting doesn't match your model")
                logger.error("  - Cache contains embeddings from a different model")
                logger.error("💡 Solution: Clear cache and regenerate with consistent model")
                return ([], [])

            # 4. Pass the embedding vector directly to the retriever.
            # The retriever expects a list of vectors.
            logger.info(f"🔍 Calling FAISS retriever with query vector of dimension {len(query_vector)}")
            logger.info(f"🔍 Retriever object type: {self.retriever.__class__.__name__}")
            logger.info(f"🔍 Retriever methods: {[m for m in dir(self.retriever) if not m.startswith('_') and callable(getattr(self.retriever, m))]}")
            
            # Try the retriever call with different approaches
            retrieved_results = None
            try:
                # Standard approach: pass list of vectors
                retrieved_results = self.retriever([query_vector])
                logger.info(f"🔍 Standard retriever call succeeded")
            except Exception as e1:
                logger.warning(f"Standard retriever call failed: {e1}")
                try:
                    # Alternative approach: pass single vector
                    retrieved_results = self.retriever(query_vector)
                    logger.info(f"🔍 Alternative retriever call (single vector) succeeded")
                except Exception as e2:
                    logger.warning(f"Alternative retriever call failed: {e2}")
                    try:
                        # Try with call method if it exists
                        if hasattr(self.retriever, 'call'):
                            retrieved_results = self.retriever.call([query_vector])
                            logger.info(f"🔍 .call() method succeeded")
                        else:
                            raise Exception("No .call() method available")
                    except Exception as e3:
                        logger.error(f"All retriever call methods failed: {e1}, {e2}, {e3}")
                        raise RuntimeError(f"FAISS retriever call failed with all methods: {e1}")
            
            if retrieved_results is not None:
                logger.info(f"🔍 Raw retriever call completed successfully")
            
            logger.info(f"📥 FAISS retriever returned: {retrieved_results.__class__.__name__}")
            logger.info(f"📊 Retrieved results length: {len(retrieved_results) if hasattr(retrieved_results, '__len__') else 'N/A'}")
            
            if not retrieved_results:
                logger.warning("Retriever returned no results for the query - this may indicate index issues")
                return ([], [])

            # Enhanced error handling for retrieval results
            if not isinstance(retrieved_results, list) or len(retrieved_results) == 0:
                logger.error(f"❌ RAG CRITICAL ERROR: Invalid retriever results format: {retrieved_results.__class__.__name__}")
                raise RuntimeError(f"RAG retriever returned invalid results: {retrieved_results.__class__.__name__}")
                
            first_result = retrieved_results[0]
            logger.info(f"🔍 First result type: {first_result.__class__.__name__ if first_result else 'None'}")
            logger.info(f"🔍 First result attributes: {dir(first_result) if first_result else 'None'}")
            
            if first_result is None:
                logger.error("❌ RAG CRITICAL ERROR: First retrieval result is None")
                raise RuntimeError("RAG retriever returned None as first result")
                
            if not hasattr(first_result, 'documents'):
                logger.error(f"❌ RAG CRITICAL ERROR: Retrieval result missing 'documents' attribute")
                logger.error(f"First result type: {first_result.__class__.__name__}")
                logger.error(f"Available attributes: {dir(first_result)}")
                raise RuntimeError(f"RAG retriever result invalid structure: {first_result.__class__.__name__}")

            # Debug the first result structure more thoroughly
            logger.info(f"🔍 DEEP DEBUG: first_result has attributes: {[attr for attr in dir(first_result) if not attr.startswith('_')]}")
            
            # Try different ways to access documents based on FAISS retriever API
            retrieved_documents = None
            if hasattr(first_result, 'documents'):
                retrieved_documents = first_result.documents
                logger.info(f"📄 Retrieved via .documents: {retrieved_documents.__class__.__name__ if retrieved_documents else 'None'}")
            elif hasattr(first_result, 'data'):
                retrieved_documents = first_result.data
                logger.info(f"📄 Retrieved via .data: {retrieved_documents.__class__.__name__ if retrieved_documents else 'None'}")
            elif hasattr(first_result, 'content'):
                retrieved_documents = first_result.content
                logger.info(f"📄 Retrieved via .content: {retrieved_documents.__class__.__name__ if retrieved_documents else 'None'}")
            else:
                # Maybe first_result IS the documents list
                if isinstance(first_result, list):
                    retrieved_documents = first_result
                    logger.info(f"📄 first_result IS the documents list: {len(retrieved_documents)} items")
                else:
                    logger.error(f"❌ Unknown retriever result structure: {first_result}")
                    logger.error(f"❌ Result value: {first_result}")
            
            logger.info(f"📄 Final retrieved documents type: {retrieved_documents.__class__.__name__ if retrieved_documents else 'None'}")
            logger.info(f"📄 Final retrieved documents length: {len(retrieved_documents) if retrieved_documents else 'None'}")
            
            if retrieved_documents is None:
                logger.error("❌ RAG CRITICAL ERROR: Retrieved documents is None after all access attempts")
                logger.error(f"❌ Original first_result: {first_result}")
                logger.error(f"❌ Available attributes: {dir(first_result)}")
                raise RuntimeError("RAG retriever returned None documents - system malfunction")
                
            if not isinstance(retrieved_documents, list):
                logger.error(f"❌ RAG CRITICAL ERROR: Retrieved documents is not a list: {retrieved_documents.__class__.__name__}")
                raise RuntimeError(f"RAG retriever returned invalid type: {retrieved_documents.__class__.__name__}")
            
            logger.info(f"✅ Successfully retrieved {len(retrieved_documents)} documents")
            return retrieved_documents, []
        except RuntimeError as e:
            # Re-raise RuntimeErrors (our critical errors) instead of masking them
            logger.error(f"❌ RAG CRITICAL ERROR: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error in RAG call: {e}", exc_info=True)
            raise RuntimeError(f"RAG system failure: {e}")
