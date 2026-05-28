import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

load_dotenv()

SCORE_THRESHOLD = 0.7


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
        self.collection_name = "movies_hybrid"

    def get_vector_store(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name="sparse",
        )

    async def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Hybrid similarity search.
        Uses similarity_search_with_relevance_scores which normalises scores to [0, 1].
        Only documents scoring above SCORE_THRESHOLD (0.7) are returned.

        Returns a list of dicts with keys:
          page_content, overview, metadata, relevance_rank, relevance_score
        """
        vector_store = self.get_vector_store()

        scored_docs = vector_store.similarity_search_with_relevance_scores(query, k=k)

        results = []
        rank = 1
        for doc, score in scored_docs:
            if score < SCORE_THRESHOLD:
                continue

            content = doc.page_content
            overview = content.split("\nOverview: ", 1)[1] if "\nOverview: " in content else ""

            results.append({
                "page_content": content,
                "overview": overview,
                "metadata": doc.metadata,
                "relevance_rank": rank,
                "relevance_score": round(score, 4),
            })
            rank += 1

        return results


qdrant_service = QdrantService()