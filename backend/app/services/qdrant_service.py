import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from app.services.logger_service import logger

load_dotenv()

SCORE_THRESHOLD = 0.65


class QdrantService:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("GEMINI_API_KEY")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = "hybrid_three_days_of_happiness"

    def get_vector_store(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="sparse",
        )

    async def search(self, query: str, k: int = 8) -> list[dict]:
        """
        Hybrid similarity search.
        Uses similarity_search_with_relevance_scores which normalises scores to [0, 1].
        Only documents scoring above SCORE_THRESHOLD (0.7) are returned.

        Returns a list of dicts with keys:
          page_content, overview, metadata, relevance_rank, relevance_score
        """
        logger.debug(f"Executing Qdrant search for query: '{query}'")
        vector_store = self.get_vector_store()

        scored_docs = vector_store.similarity_search_with_relevance_scores(query, k=k)

        results = []
        rank = 1
        for doc, score in scored_docs:
            if score < SCORE_THRESHOLD:
                logger.debug(f"Filtered out chunk (Score: {score:.4f} < Threshold: {SCORE_THRESHOLD})")
                continue

            logger.debug(f"Retrieved chunk (Score: {score:.4f}) | Preview: {doc.page_content[:50]}...")
            content = doc.page_content

            results.append({
                "page_content": content,
                "metadata": doc.metadata,
                "relevance_rank": rank,
                "relevance_score": round(score, 4),
            })
            rank += 1

        logger.info(f"Qdrant retrieved {len(results)} valid chunks.")
        return results
    
    def verify_hybrid_setup(self):
        """Pulls one record to prove sparse vectors actually exist in the DB."""
        from app.services.logger_service import logger
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_vectors=True
            )
            if records:
                vectors = records[0].vector

                is_dict = isinstance(vectors, dict)
                has_sparse = is_dict and "sparse" in vectors
                
                logger.info("--- Qdrant Hybrid Check ---")
                logger.info(f"Vector format is dictionary (Named Vectors): {is_dict}")
                if is_dict:
                    logger.info(f"Found Vector Names: {list(vectors.keys())}")
                    logger.info(f"BM25 Sparse Vectors Present: {'YES' if has_sparse else 'NO'}")
            else:
                logger.warning("Qdrant collection is completely empty!")
        except Exception as e:
            logger.error(f"Failed to verify Qdrant setup: {e}")


qdrant_service = QdrantService()