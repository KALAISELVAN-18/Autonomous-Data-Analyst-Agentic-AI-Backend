from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.config import settings


def get_llm(temperature: float = 0.2, model: str = None) -> Any:
    """
    Returns a configured Gemini LLM instance with flash fallback.
    Temperature is kept low by default for data analysis tasks
    to ensure deterministic, accurate code and analytical output.
    """
    model_name = model or settings.gemini_model or "gemini-2.5-pro"
    
    main_llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
        max_retries=1,
    )
    
    if "pro" in model_name.lower():
        fallback_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.gemini_api_key,
            temperature=temperature,
        )
        return main_llm.with_fallbacks([fallback_llm])
        
    return main_llm


