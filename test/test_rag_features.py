import unittest
from unittest.mock import patch, MagicMock
from api.rag import RAG
from api.config import configs

class TestRAGFeatures(unittest.TestCase):

    @patch('api.rag.get_embedder')
    def test_reranking(self, mock_get_embedder):
        # Arrange
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_api_key'}):
            rag = RAG()
            doc1 = MagicMock()
            doc1.text = "this is a test document"
            doc2 = MagicMock()
            doc2.text = "this is another test document"
            doc3 = MagicMock()
            doc3.text = "this is a third test document with the keyword python"
            documents = [doc1, doc2, doc3]

            # Act
            reranked_documents = rag.rerank_documents("python", documents)

            # Assert
            self.assertEqual(reranked_documents, [doc3, doc1, doc2])

if __name__ == '__main__':
    unittest.main()
