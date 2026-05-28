from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config.setting import EMBEDDING_MODEL

class EmbeddingManager:
    """
    Handles embedding generation and ChromaDB storage.
    """

    def __init__(
        self,
        persist_directory="database/chroma_db"
    ):

        self.persist_directory = persist_directory

        # Embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        self.vector_store = None

    def create_vector_store(self, chunks):
        """
        Create ChromaDB vector database from chunks.

        Args:
            chunks (list): List of text chunks
        """

        self.vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )

        return self.vector_store

    def load_vector_store(self):
        """
        Load existing ChromaDB database.
        """

        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

        return self.vector_store

    def similarity_search(self, query, k=4):
        """
        Perform semantic similarity search.

        Args:
            query (str): User query
            k (int): Number of results

        Returns:
            list: Retrieved documents
        """

        results = self.vector_store.similarity_search(
            query,
            k=k
        )

        return results