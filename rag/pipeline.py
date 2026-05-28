import os
import shutil

from utils.pdf_loader import PDFLoader
from utils.chunking import TextChunker
from utils.embeddings import EmbeddingManager

from rag.retriever import HybridRetriever
from rag.query_classifier import QueryClassifier
from rag.generator import ResponseGenerator


class AdvancedRAG:
    """
    Main Advanced RAG Pipeline
    """

    def __init__(self):

        # ---------------------------------------------
        # PDF Processing
        # ---------------------------------------------

        self.pdf_loader = PDFLoader()

        self.chunker = TextChunker()

        # ---------------------------------------------
        # Embeddings + Vector DB
        # ---------------------------------------------

        self.embedding_manager = EmbeddingManager()

        # ---------------------------------------------
        # Query Handling
        # ---------------------------------------------

        self.query_classifier = QueryClassifier()

        # ---------------------------------------------
        # Response Generation
        # ---------------------------------------------

        self.response_generator = ResponseGenerator()

        # ---------------------------------------------
        # Runtime Variables
        # ---------------------------------------------

        self.retriever = None

        self.vector_store = None

        self.chunks = []

    # =================================================
    # PROCESS DOCUMENT
    # =================================================

    def process_document(self, uploaded_file):
        """
        Process uploaded PDF document.
        """

        # ---------------------------------------------
        # Reset Previous References
        # ---------------------------------------------

        self.retriever = None

        self.vector_store = None

        self.chunks = []

        # ---------------------------------------------
        # Clear Previous Chroma Database
        # ---------------------------------------------

        try:

            if os.path.exists("database/chroma_db"):

                shutil.rmtree(
                    "database/chroma_db"
                )

        except PermissionError:

            pass

        # ---------------------------------------------
        # Extract Text From PDF
        # ---------------------------------------------

        text = self.pdf_loader.extract_text(
            uploaded_file
        )

        # ---------------------------------------------
        # Chunk Text
        # ---------------------------------------------

        self.chunks = self.chunker.split_text(
            text
        )

        # ---------------------------------------------
        # Create Vector Database
        # ---------------------------------------------

        self.vector_store = (
            self.embedding_manager.create_vector_store(
                self.chunks
            )
        )

        # ---------------------------------------------
        # Initialize Retriever
        # ---------------------------------------------

        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            chunks=self.chunks
        )

    # =================================================
    # QUESTION ANSWERING PIPELINE
    # =================================================

    def ask_question(
        self,
        query,
        chat_history=""
    ):
        """
        Main question-answering pipeline.
        """

        # ---------------------------------------------
        # Safety Check
        # ---------------------------------------------

        if self.retriever is None:

            return {
                "query_type": "none",
                "context": [],
                "response": (
                    "Please upload and process "
                    "a PDF document first."
                )
            }

        # ---------------------------------------------
        # Step 1: Query Classification
        # ---------------------------------------------

        query_type = self.query_classifier.classify(
            query
        )

        # ---------------------------------------------
        # Step 2: Context Retrieval
        # ---------------------------------------------

        if query_type == "summary":

            # Faster Summary Retrieval
            context = self.chunks[:5]

        else:

            context = self.retriever.hybrid_search(
                query=query,
                k=3
            )

        # ---------------------------------------------
        # Step 3: Generate Response
        # ---------------------------------------------

        response = (
            self.response_generator.generate_response(
                query=query,
                context=context,
                query_type=query_type,
                chat_history=chat_history
            )
        )

        # ---------------------------------------------
        # Final Output
        # ---------------------------------------------

        return {
            "query_type": query_type,
            "context": context,
            "response": response
        }