from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    This class defines the exact environment variables our app needs.
    If any of these are missing in the .env file, the app will crash on startup,
    saving us from silent errors!
    """
    OPENAI_API_KEY: str = Field(min_length=1)
    PINECONE_API_KEY: str = Field(min_length=1)
    PINECONE_INDEX_NAME: str = "financial-rag"

    # Redis Semantic Cache (optional — app works without it)
    REDIS_HOST:     str = ""
    REDIS_PORT:     int = 6379
    REDIS_USERNAME: str = "default"
    REDIS_PASSWORD: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# We instantiate a single instance of the settings to be imported anywhere in our app
settings = Settings()
