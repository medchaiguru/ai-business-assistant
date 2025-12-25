"""Settings"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    # === App Settings ===
    app_name: str = "AI Customer Support Automation System"
    debug: bool = True
    version: str = "0.1.0"

    # === API Keys / External Services ===
    OPENAI_API_KEY: SecretStr | None = None

    # === LangSmith Tracing ===
    # Enable tracing (default to False so it doesn't break if not configured)
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_ENDPOINT: str = "https://eu.api.smith.langchain.com"
    LANGCHAIN_API_KEY: SecretStr | None = None
    LANGCHAIN_PROJECT: str = "ai-customer-support"

    # default folder to save crawled data
    CRAWLED_DATA_DIR: str = "data"

    # Source data
    DATA_PATH: str = "src/Sinbi Muay Thai_data.json"

    # Path where the Chroma index will be saved (DEV mode)
    DATA_VECTOR_PATH: str = "data/chroma_index"
    SEMANTIC_CACHE_PATH: str = "data/semantic_cache"

    # === ChromaDB Server Settings ===
    CHROMA_SERVER_HOST: str = "localhost"
    CHROMA_SERVER_HTTP_PORT: int = 8000

    # Mapping of URLs to page content type / page name
    URL_PAGE_MAP: dict[str, str] = {
        "https://www.sinbimuaythai.com/": "Home",
        "https://www.sinbimuaythai.com/meet-the-team/": "Team",
        "https://www.sinbimuaythai.com/training-prices/": "Pricing",
        "https://www.sinbimuaythai.com/accommodation/": "Accommodation",
        "https://www.sinbimuaythai.com/faq/": "FAQ",
        "https://www.sinbimuaythai.com/muay-thai-fights-phuket/": "Fights",
        "https://www.sinbimuaythai.com/discord/": "Discord",
    }

    # Fast API Endpoint
    QUERY_URL: str = "http://127.0.0.1:8000/query"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Singleton instanceP
settings = Settings()
