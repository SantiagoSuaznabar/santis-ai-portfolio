import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key=api_key,
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.output_parser = StrOutputParser()

    async def get_simple_response(self, prompt: str) -> str:
        """Prueba básica para asegurar la conexión"""
        messages = [
            SystemMessage(content="Eres un AI Engineer experto. Responde de forma clara y técnica."),
            HumanMessage(content=prompt)
        ]
        
        chain = self.llm | self.output_parser
        response_text = await chain.ainvoke(messages)
        return response_text

llm_service = LLMService()