from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:
    """
    Advanced Hybrid Retriever

    Combines:
    1. Semantic Search (Vector Search)
    2. BM25 Keyword Search

    Also filters irrelevant queries using similarity thresholds.
    """

    def __init__(self, vector_store, chunks):

        self.vector_store = vector_store
        self.chunks = chunks

        # Tokenize chunks for BM25
        tokenized_chunks = [
            chunk.split() for chunk in chunks
        ]

        self.bm25 = BM25Okapi(tokenized_chunks)

    def vector_search(self, query, k=4):
        """
        Semantic similarity search with filtering.
        """

        results = self.vector_store.similarity_search_with_score(
            query,
            k=k
        )

        filtered_results = []

        for doc, score in results:

            # Lower score = better match
            # Adjust threshold if needed
            if score < 1.5:
                filtered_results.append(
                    doc.page_content
                )

        return filtered_results

    def keyword_search(self, query, k=4):
        """
        BM25 keyword search.
        """

        tokenized_query = query.split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        # Get top matching chunks
        top_indices = np.argsort(scores)[-k:]

        results = []

        for i in top_indices:

            # Ignore zero-score matches
            if scores[i] > 0:
                results.append(
                    self.chunks[i]
                )

        return results

    def hybrid_search(self, query, k=4):
        """
        Combine vector + BM25 retrieval.
        """

        vector_results = self.vector_search(
            query,
            k
        )

        keyword_results = self.keyword_search(
            query,
            k
        )

        # Combine + remove duplicates
        combined_results = list(
            set(vector_results + keyword_results)
        )

        return combined_results[:k]