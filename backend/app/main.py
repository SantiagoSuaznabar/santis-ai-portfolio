from fastapi import FastAPI
from dotenv import load_dotenv
import os

from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm_service import llm_service
from app.services.qdrant_service import qdrant_service

# Load ENV variables
load_dotenv()

app = FastAPI(
    title="AI Portfolio Backend",
    description="Central API to handle RAG and Agents with LangGraph",
    version="1.0.0"
)

# Test Endpoint/Health Check
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "AI Backend Online. Ready to receive petitions",
        "environment": os.getenv("ENV_MODE", "development")
    }

@app.post("/api/chat/test", response_model=ChatResponse)
async def test_chat(request: ChatRequest):
    """
    Test Endpoint to verify connection to Gemini through LangChain.
    """
    
    ai_response = await llm_service.get_simple_response(request.message)
    
    return ChatResponse(response=ai_response)