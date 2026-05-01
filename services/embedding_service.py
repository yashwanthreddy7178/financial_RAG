from openai import AsyncOpenAI
from config.settings import settings

class EmbeddingService:
    def __init__(self):
        """
        We initialize the official OpenAI client using our Pydantic settings.
        Notice we don't hardcode the key here!
        """
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # 'text-embedding-3-small' is currently OpenAI's best balance of cheap, fast, and highly accurate
        self.model = "text-embedding-3-small"

    async def generate_embeddings(self, chunks: list[str]) -> list[list[float]]:
        """
        Takes a list of text chunks and asks OpenAI to convert them into
        a list of 1536-dimensional float arrays (Vectors).
        """
        if not chunks:
            return []
            
        # We call the OpenAI API. It is smart enough to process a whole batch of chunks at once!
        response = await self.client.embeddings.create(
            input=chunks,
            model=self.model
        )
        
        # The response payload from OpenAI has extra metadata we don't need right now.
        # We just loop through the response and extract the raw array of 1,536 numbers.
        embeddings = [item.embedding for item in response.data]
        
        return embeddings
