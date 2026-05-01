from flashrank import Ranker, RerankRequest

class RerankerService:
    def __init__(self):
        """
        Initializes the Cross-Encoder.
        We use ms-marco-MiniLM-L-12-v2 because it is highly accurate but small enough 
        to run instantly on a CPU without requiring a massive GPU.
        """
        # The very first time this runs, it will download a ~100MB model to your computer.
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="flashrank_cache")

    def rerank_chunks(self, query: str, chunks: list[str], top_k: int = 5) -> list[str]:
        """
        Takes the query and a large list of chunks (e.g., 20 chunks),
        uses the Cross-Encoder to deeply read them, and returns the top_k best chunks.
        """
        if not chunks:
            return []
            
        # FlashRank expects the data in a specific dictionary format
        passages = []
        for i, chunk in enumerate(chunks):
            passages.append({
                "id": str(i),
                "text": chunk
            })
            
        # We package the query and the passages together
        rerankrequest = RerankRequest(query=query, passages=passages)
        
        # The Cross-Encoder reads them together and scores them!
        results = self.ranker.rerank(rerankrequest)
        
        # The results are sorted from best (highest score) to worst.
        # We slice the list to only keep the top_k results.
        top_results = results[:top_k]
        
        # Extract just the raw text back out to send to the LLM
        best_chunks = [result["text"] for result in top_results]
        
        return best_chunks
