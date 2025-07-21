import unittest
from unittest.mock import patch, MagicMock
from api.rag import RAG
from api.config import configs

class TestRAGFeatures(unittest.TestCase):

    @patch('api.data_pipeline.summarize_large_file')
    @patch('api.rag.RAG.rerank_documents')
    @patch('api.rag.FAISSRetriever')
    @patch('api.rag.get_embedder')
    @patch('api.rag.DatabaseManager')
    def test_summarization_and_reranking(self, mock_db_manager, mock_get_embedder, mock_faiss_retriever, mock_rerank_documents, mock_summarize_large_file):
        # Arrange
        with patch.dict(configs, {"repository": {"rerank_documents": True, "max_summarization_tokens": 100}}):
            with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_api_key'}):
                rag = RAG()
                rag.retriever = MagicMock()

                # Act
                rag.call("test query")

                # Assert
                mock_rerank_documents.assert_called_once()

if __name__ == '__main__':
    unittest.main()
