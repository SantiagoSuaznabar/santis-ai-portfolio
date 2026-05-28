import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

class QdrantService:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        api_key = os.getenv("GEMINI_API_KEY")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )
        
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = "portfolio_rag"

        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Verify if collection exists in Qdrant, if not, it initializes it"""
        collections_response = self.client.get_collections()
        collections = [col.name for col in collections_response.collections]
        
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                ),
            )
            print(f"Collection '{self.collection_name}' created succesfully")
        else:
            print(f"Collection '{self.collection_name}' already exists, ready to use")

    def get_vector_store(self):
        """"""
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

qdrant_service = QdrantService()