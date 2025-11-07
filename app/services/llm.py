from langchain_openai import ChatOpenAI

from app.core.config import settings
llm_model = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.2,
    max_tokens=512,
    openai_api_key=settings.OPENAI_API_KEY
)
