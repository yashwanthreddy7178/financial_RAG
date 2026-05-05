from openai import AsyncOpenAI, RateLimitError
from config.settings import settings
import asyncio
import logging
import random

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------
async def _with_backoff(coro_fn, max_retries: int = 6, base_delay: float = 1.0):
    """
    Calls an async callable with exponential backoff + jitter on RateLimitError.
    max_retries=6 gives waits of ~1s, 2s, 4s, 8s, 16s, 32s before giving up.
    """
    for attempt in range(max_retries):
        try:
            return await coro_fn()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise  # exhausted all retries
            wait = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                "Rate limit hit (attempt %d/%d). Retrying in %.1fs… [%s]",
                attempt + 1, max_retries, wait, e,
            )
            await asyncio.sleep(wait)


class LLMService:
    def __init__(self):
        """
        Initializes the OpenAI client for chat completion.
        """
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        # We use gpt-4o-mini because it is incredibly fast, very cheap, and highly capable for RAG.
        self.model = "gpt-4o-mini"

    async def generate_answer(self, question: str, context_chunks: list[dict], conversation_history: list[dict] = None) -> str:
        """
        Takes the user's question, the context chunks retrieved from Pinecone,
        and the conversation history, and asks the LLM to synthesize a final answer.
        """
        # Format the context chunks to include source information
        formatted_chunks = []
        for i, doc in enumerate(context_chunks):
            # We use 1-indexed citations for user-friendliness
            formatted_chunks.append(f"[Source {i+1}: {doc['filename']}]\n{doc['text']}")
            
        context_text = "\n\n---\n\n".join(formatted_chunks)
        
        system_prompt = f"""
        You are a highly intelligent and precise financial analyst AI.
        You must answer the user's question using ONLY the provided context below.
        You also have access to the conversation history to understand follow-up questions.
        
        If the exact or complete answer cannot be found in the context, you should gracefully explain what information is missing, but still provide any relevant related information that IS present in the context.
        For example: "The provided documents do not specify the exact [metric]. However, the report notes that..."
        
        CRITICAL RULE: Do not use your own outside knowledge. If the context contains absolutely nothing related to the query, state clearly that the document does not contain the information.

        CITATION RULE: You MUST cite your sources inline using the provided Source IDs in brackets. 
        For example: "The company's revenue grew by 15% [1]." 
        Only use the numerical ID in brackets, not the filename. Do not write [Source 1], just [1].
        
        CONTEXT:
        {context_text}
        """
        
        # Build the messages array starting with the system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        # Inject the conversation history (old Q&A pairs) into the middle
        # This is the entire secret of how memory works!
        if conversation_history:
            messages.extend(conversation_history)
        
        # Finally, append the current question at the very end
        messages.append({"role": "user", "content": question})
        
        response = await _with_backoff(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
        )
        
        # We extract and return just the text of the AI's answer
        return response.choices[0].message.content

    async def route_query(self, question: str) -> str:
        """
        The Traffic Cop! Classifies the user's intent before we waste resources on the 
        full RAG pipeline. Returns one of three strings: 'rag', 'small_talk', 'off_topic'.
        """
        system_prompt = """
        You are an intent classification engine for a financial document Q&A system.
        Your only job is to classify the user's message into exactly ONE of the following categories:

        - "rag": The user is asking a question about financial documents, data, reports, companies, earnings, risks, revenue, or any business/financial topic.
        - "small_talk": The user is greeting, thanking, or having a casual conversation (e.g., "Hello", "Thanks", "Who are you?").
        - "off_topic": The user is asking about something completely unrelated to finance or this application (e.g., "Write me a poem", "What is the weather?").

        CRITICAL RULE: Your response must be ONLY one of the three words: rag, small_talk, or off_topic.
        Do not add any punctuation, explanation, or extra text. Just the single word.
        """

        response = await _with_backoff(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0,      # Zero creativity — we want perfectly deterministic routing!
                max_tokens=5        # The answer is just one word, so we cap tokens aggressively
            )
        )

        # Extract the route, strip any accidental whitespace, and lowercase it
        route = response.choices[0].message.content.strip().lower()

        # Defensive check: if the LLM returns something unexpected, default to 'rag'
        if route not in ["rag", "small_talk", "off_topic"]:
            return "rag"

        return route

    async def generate_small_talk(self, question: str, conversation_history: list[dict] = None) -> str:
        """
        Handles casual conversation without accessing any documents.
        """
        system_prompt = """
        You are a friendly and professional financial AI assistant called "Finance AI".
        You are having a casual conversation with the user. 
        You can answer general questions about who you are and what you do, but remind 
        users that your primary purpose is to answer questions about uploaded financial documents.
        """
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": question})

        response = await _with_backoff(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
        )
        return response.choices[0].message.content

    async def expand_query(
        self,
        original_query: str,
        conversation_history: list[dict] | None = None
    ) -> list[str]:
        """
        Takes a user's query and generates 3 descriptive search variations.
        If conversation_history is provided and the query contains reference pronouns
        (it, its, they, this, etc.), the query is first rewritten to resolve the
        pronoun using the conversation context before expansion.
        This fixes retrieval for follow-up questions like "how is it affecting the market?"
        """
        PRONOUNS = {"it", "its", "they", "their", "this", "that", "these", "those", "he", "she", "his", "her"}
        query_words = set(original_query.lower().split())
        has_pronoun = bool(query_words & PRONOUNS)

        # Step 1: If the question has a pronoun and we have history, resolve it first
        resolved_query = original_query
        if has_pronoun and conversation_history:
            rewrite_prompt = """
            You are a query rewriter. The user has asked a follow-up question that contains a pronoun (it, its, they, this, etc.).
            Using the conversation history, rewrite the question to replace all pronouns with the specific entity they refer to.
            Return ONLY the rewritten question. No explanation, no quotes.

            Example:
            History: User asked about NVIDIA. 
            Follow-up: "How is it affecting the market?"
            Rewritten: "How is NVIDIA affecting the stock market?"
            """
            messages = [{"role": "system", "content": rewrite_prompt}]
            messages.extend(conversation_history[-4:])  # Only last 2 exchanges for efficiency
            messages.append({"role": "user", "content": f'Rewrite this question: "{original_query}"'})

            rewrite_response = await _with_backoff(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0  # Deterministic — we want precision here
                )
            )
            resolved_query = rewrite_response.choices[0].message.content.strip().strip('"')
            logger.info(f"Pronoun resolved: '{original_query}' → '{resolved_query}'")

        # Step 2: Expand the (now resolved) query into 3 search variations
        system_prompt = """
        You are a financial query expansion engine.
        Your job is to take a user's search query and write 3 different, highly descriptive variations of it.
        This is for a semantic search engine. The variations should cover different synonyms, perspectives, and potential phrasings a financial document might use.

        CRITICAL RULE: Return ONLY the 3 queries, separated by newlines. No numbers, no bullet points, no introductory text.
        """

        response = await _with_backoff(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": resolved_query}
                ],
                temperature=0.7
            )
        )

        raw_output = response.choices[0].message.content
        queries = [q.replace("- ", "").strip() for q in raw_output.split('\n') if q.strip()]

        # Return the resolved query (not the original pronoun version) + up to 3 variations
        return [resolved_query] + queries[:3]

    async def contextualize_chunks(self, full_text: str, chunks: list[str]) -> list[str]:
        """
        Implements Contextual Retrieval (Anthropic, 2024).
        For each chunk, uses a sliding window (document prefix + local context)
        to generate a 2-sentence context summary and prepends it to the chunk.
        
        Uses a Semaphore to cap concurrency at 3 simultaneous OpenAI calls to stay
        well under the 200k TPM limit. Chunks are processed in batches of 3 with a
        1-second pause between batches to avoid sustained rate-limit pressure.
        """
        # Document prefix: identifies the document globally (company, year, filing type)
        document_prefix = full_text[:3000]
        
        # Semaphore = a traffic gate that only allows 3 LLM calls at the same time.
        # 3 concurrent calls × ~600 tokens each ≈ 1,800 tokens/batch — very safe.
        semaphore = asyncio.Semaphore(3)
        
        async def _contextualize_one(chunk: str) -> str:
            """Generates context for a single chunk and prepends it, with backoff retry."""
            async with semaphore:
                # SLIDING WINDOW: find where this chunk sits in the full document
                # and grab the 500 characters immediately before it for local context
                chunk_pos = full_text.find(chunk[:80]) # Use first 80 chars to locate the chunk
                if chunk_pos > 0:
                    local_context = full_text[max(0, chunk_pos - 500) : chunk_pos]
                else:
                    local_context = ""
                
                prompt = f"""Here is the beginning of a financial document:
<document_start>
{document_prefix}
</document_start>

Here is the text that appears immediately before the chunk (local context):
<local_context>
{local_context}
</local_context>

Here is the specific chunk to contextualize:
<chunk>
{chunk}
</chunk>

Write a brief 1-2 sentence context summary that identifies what document this is from,
what section it belongs to, and what topic it covers. This will be prepended to the chunk
to improve semantic search retrieval. Be concise and factual."""

                response = await _with_backoff(
                    lambda p=prompt: self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": p}],
                        max_tokens=120,  # Context summary should be short!
                        temperature=0
                    )
                )
                
                context_summary = response.choices[0].message.content.strip()
                
                # Prepend the context to the chunk with a clear separator
                return f"[CONTEXT: {context_summary}]\n\n{chunk}"
        
        # -----------------------------------------------------------------------
        # Process in micro-batches of 3 with a 1-second cooldown between batches.
        # This ensures we never sustain more than ~5,400 tokens/min on this path.
        # -----------------------------------------------------------------------
        BATCH_SIZE = 3
        logger.info("Contextualizing %d chunks (batch_size=%d, cooldown=1s)…", len(chunks), BATCH_SIZE)
        print(f"  Contextualizing {len(chunks)} chunks (3 concurrent, 1s cooldown between batches)...")

        contextualized_chunks: list[str] = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            batch_results = await asyncio.gather(*[_contextualize_one(c) for c in batch])
            contextualized_chunks.extend(batch_results)
            # Cooldown: skip the pause after the very last batch
            if i + BATCH_SIZE < len(chunks):
                await asyncio.sleep(1.0)
        
        return contextualized_chunks

    async def grade_context(self, question: str, chunks: list[dict]) -> str:
        """
        The Self-RAG Grader. Reads the retrieved chunks alongside the question and
        returns 'sufficient' if they contain enough information to answer,
        or 'insufficient' if they are too vague or irrelevant.
        """
        # We join only the first 3 chunks for grading — reading all 5 is overkill
        # and wastes tokens. If the top 3 are bad, all 5 are bad.
        context_preview = "\n\n---\n\n".join([c["text"] for c in chunks[:3]])

        system_prompt = """
        You are a strict retrieval quality grader for a financial RAG system.
        You will be given a user question and retrieved document passages.
        
        Your only job is to decide if the passages contain enough relevant information 
        to meaningfully answer the question.
        
        Grade as 'sufficient' if the passages contain direct, relevant facts about the question topic.
        Grade as 'insufficient' if the passages are vague, off-topic, or clearly missing key information.
        
        CRITICAL RULE: Respond with ONLY one word — either: sufficient or insufficient
        """

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nRetrieved Passages:\n{context_preview}"}
            ],
            temperature=0,    # Must be deterministic
            max_tokens=5      # Only need one word
        )

        grade = response.choices[0].message.content.strip().lower()

        # Defensive check: if LLM returns something unexpected, treat as sufficient
        # (We don't want a grading bug to break the entire pipeline)
        if grade not in ["sufficient", "insufficient"]:
            return "sufficient"

        return grade

    async def rewrite_query(self, original_question: str) -> str:
        """
        The Self-RAG Query Rewriter. Called when the grader returns 'insufficient'.
        Rewrites the query using different vocabulary so the second Pinecone search
        finds different (hopefully better) chunks.
        """
        system_prompt = """
        You are a search query optimization expert for a financial document RAG system.
        
        A user's query was used to search a vector database but the retrieved documents
        were not relevant enough to answer the question.
        
        Rewrite the query to be more specific, use different financial vocabulary,
        and try alternative phrasings that might surface more relevant document passages.
        
        Return ONLY the rewritten query. No explanation, no prefix like "Rewritten query:".
        """

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Original query that failed: {original_question}"}
            ],
            temperature=0.3,  # Small amount of creativity for different vocabulary
            max_tokens=100
        )

        return response.choices[0].message.content.strip()
