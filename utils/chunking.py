from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.setting import CHUNK_SIZE, CHUNK_OVERLAP

class TextChunker:
    """
    Splits document text into smaller chunks.
    """

    def __init__(
        self,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    ):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text):
        """
        Split text into chunks.

        Args:
            text (str): Full document text

        Returns:
            list: Text chunks
        """

        chunks = self.text_splitter.split_text(text)

        return chunks