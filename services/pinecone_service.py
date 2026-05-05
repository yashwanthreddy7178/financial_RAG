from pinecone import Pinecone, ServerlessSpec
from config.settings import settings
import time

class PineconeService:
    def __init__(self):
        """
        Initializes the official Pinecone client using our secure API key.
        """
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        
        # We will hold our connection to the specific database table (index) here
        self.index = None

    def ensure_index_exists(self):
        """
        Checks if our vector database index exists. If not, it creates it!
        """
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index '{self.index_name}'... This can take ~60 seconds.")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536, # CRUCIAL: This MUST match OpenAI's output size of 1536!
                metric="cosine", # The mathematical formula used to find "closest" vectors
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Pinecone takes a little bit of time to boot up a new database on AWS
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)

        # Connect to the live index
        self.index = self.pc.Index(self.index_name)

    def upsert_vectors(self, filename: str, chunks: list[str], embeddings: list[list[float]]):
        """
        Uploads our numbers and text to the Pinecone database.
        'Upsert' means "Update or Insert".
        """
        vectors_to_upload = []
        
        for i, chunk in enumerate(chunks):
            # 1. Create a unique ID for every single paragraph
            unique_id = f"{filename}_chunk_{i}"
            
            # 2. Package it exactly how Pinecone expects it
            vector_package = {
                "id": unique_id,
                "values": embeddings[i], # The 1,536 numbers
                "metadata": {
                    "text": chunk,           # The actual readable paragraph
                    "filename": filename     # Metadata so we can filter by filename later!
                }
            }
            vectors_to_upload.append(vector_package)
            
        # 3. Upload them in batches of 100!
        # Pinecone has a strict 4MB payload limit per request. 
        # If we upload a massive 10-K report (1000+ chunks), we will crash the server.
        batch_size = 100
        for i in range(0, len(vectors_to_upload), batch_size):
            batch = vectors_to_upload[i:i + batch_size]
            self.index.upsert(vectors=batch)

    def search_vectors(self, question_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Searches the Pinecone database for the most mathematically similar vectors.
        Returns a list of dictionaries containing the chunk text, filename, and ID.
        """
        # We ask Pinecone to do the complex Cosine Similarity math for us!
        search_results = self.index.query(
            vector=question_embedding,
            top_k=top_k,               # How many chunks to return (default 5)
            include_metadata=True      # CRUCIAL: We need the actual text back, not just the numbers!
        )
        
        # Pinecone returns a complex JSON object. We loop through the matches
        # and extract the English text and metadata we saved earlier.
        retrieved_chunks = []
        for match in search_results.matches:
            chunk_data = {
                "id": match.id,
                "text": match.metadata["text"],
                "filename": match.metadata.get("filename", "Unknown Source")
            }
            retrieved_chunks.append(chunk_data)
            
        return retrieved_chunks
