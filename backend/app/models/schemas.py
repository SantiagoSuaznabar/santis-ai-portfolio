from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="Users message or question", min_length=1)

class ChatResponse(BaseModel):
    response: str