from fastapi import FastAPI
from dotenv import load_dotenv
import os

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