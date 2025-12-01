from langchain_openai import ChatOpenAI

from app.config import settings


llm_model = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.2,
    max_tokens=512,
    openai_api_key=settings.OPENAI_API_KEY
)

def llm_cost(input_tokens: int, output_tokens: int) -> float:
    # Pricing (USD per token)
    GPT4O_MINI_INPUT_PRICING = 0.15 / 1_000_000
    GPT4O_MINI_OUTPUT_PRICING = 0.60 / 1_000_000

    cost = (
        input_tokens * GPT4O_MINI_INPUT_PRICING +
        output_tokens * GPT4O_MINI_OUTPUT_PRICING
    )
    return cost