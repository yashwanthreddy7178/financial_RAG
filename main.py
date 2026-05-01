import logging
import os
import shutil
import uuid

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel

from config.settings import settings
from services.document_processor import DocumentProcessor
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.pinecone_service import PineconeService
from services.reranker_service import RerankerService
from services.semantic_cache import SemanticCache

# ─── Structured Logging ───────────────────────────────────────────────────────
# Replace all print() calls with a proper logger.
# In production, these logs are collected by services like Datadog or CloudWatch.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("financial_rag")

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ─── Rate Limiter ─────────────────────────────────────────────────────────────
# slowapi uses the client's IP address as the rate limit key.
limiter = Limiter(key_func=get_remote_address)

# ─── Lazy Service Singletons ──────────────────────────────────────────────────
processor     = None
embedding_svc = None
pinecone_svc  = None
llm_svc       = None
reranker_svc  = None

# Semantic cache is initialised once at module load — fail-safe, never crashes
semantic_cache = SemanticCache(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    username=settings.REDIS_USERNAME,
    password=settings.REDIS_PASSWORD,
)


# ─── Pydantic Models ──────────────────────────────────────────────────────────
class SearchQuery(BaseModel):
    question: str
    conversation_history: list[dict] | None = None

class TestCase(BaseModel):
    question: str
    ground_truth: str

class EvaluationRequest(BaseModel):
    test_cases: list[TestCase]


# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Financial RAG API",
    description="A production-grade AI-powered search engine for financial documents.",
    version="2.0.0",
)

# Enable CORS for potential future separated frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the rate limiter and its error handler with FastAPI
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── Helper: Consistent Error Response ────────────────────────────────────────
def error_response(status_code: int, message: str):
    """Returns a consistent JSON error response across all endpoints."""
    raise HTTPException(status_code=status_code, detail={"error": message})


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    return {"status": "online", "version": "2.0.0"}


@app.post("/ingest")
@limiter.limit("5/minute")  # Max 5 document uploads per IP per minute
async def ingest_document(request: Request, file: UploadFile = File(...)):
    """
    Receives a PDF file, processes it with Contextual Retrieval,
    and stores the enriched vectors in Pinecone.
    """
    # ── File Validation & Streaming ───────────────────────────────────────────
    if not file.filename.lower().endswith(".pdf"):
        error_response(400, "Only PDF files are accepted.")

    # Prevent Path Traversal and Name Collisions by using a UUID
    safe_filename = f"{uuid.uuid4()}.pdf"
    temp_file_path = f"temp_{safe_filename}"

    try:
        # Prevent memory exhaustion (DoS) by reading the file in chunks
        size = 0
        with open(temp_file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                size += len(chunk)
                if size > MAX_FILE_SIZE_BYTES:
                    buffer.close()
                    os.remove(temp_file_path)
                    error_response(413, f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB.")
                buffer.write(chunk)

        # ── Lazy-load services ────────────────────────────────────────────────
        global processor, embedding_svc, pinecone_svc, llm_svc
        if processor is None:
            processor = DocumentProcessor()
        if embedding_svc is None:
            embedding_svc = EmbeddingService()
        if pinecone_svc is None:
            pinecone_svc = PineconeService()
            pinecone_svc.ensure_index_exists()
        if llm_svc is None:
            llm_svc = LLMService()

        # ── Pipeline ──────────────────────────────────────────────────────────
        logger.info(f"Ingestion started | file={file.filename}")

        full_text, chunks = processor.process_pdf(temp_file_path)
        logger.info(f"Chunking complete | file={file.filename} | chunks={len(chunks)}")

        contextual_chunks = await llm_svc.contextualize_chunks(full_text, chunks)
        logger.info(f"Contextual Retrieval complete | enriched_chunks={len(contextual_chunks)}")

        embeddings = await embedding_svc.generate_embeddings(contextual_chunks)
        logger.info(f"Embeddings generated | count={len(embeddings)}")

        pinecone_svc.upsert_vectors(file.filename, contextual_chunks, embeddings)
        logger.info(f"Ingestion complete | file={file.filename}")

        return {
            "message": "Document successfully ingested with Contextual Retrieval.",
            "filename": file.filename,
            "total_chunks": len(contextual_chunks),
        }

    except Exception as e:
        logger.error(f"Ingestion failed | file={file.filename} | error={str(e)}")
        error_response(500, f"Ingestion failed: {str(e)}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/search")
@limiter.limit("30/minute")  # Max 30 searches per IP per minute
async def search_documents(request: Request, query: SearchQuery):
    """
    Routes the query, expands it, retrieves context with Self-RAG,
    and returns a grounded answer.
    """
    if not query.question.strip():
        error_response(400, "Question cannot be empty.")

    # ── Lazy-load services ────────────────────────────────────────────────────
    global embedding_svc, pinecone_svc, llm_svc, reranker_svc
    if embedding_svc is None:
        embedding_svc = EmbeddingService()
    if pinecone_svc is None:
        pinecone_svc = PineconeService()
        pinecone_svc.ensure_index_exists()
    if reranker_svc is None:
        reranker_svc = RerankerService()
    if llm_svc is None:
        llm_svc = LLMService()

    logger.info(f"Search started | question={query.question[:80]}")

    # ── Stage 0: Query Routing ─────────────────────────────────────────────────
    route = await llm_svc.route_query(query.question)
    logger.info(f"Route decision | route={route}")

    if route == "small_talk":
        answer = await llm_svc.generate_small_talk(
            question=query.question,
            conversation_history=query.conversation_history,
        )
        return {"question": query.question, "answer": answer, "sources_used": 0}

    if route == "off_topic":
        return {
            "question": query.question,
            "answer": "I'm a specialized financial document assistant. I can only answer questions about uploaded financial reports. Please upload a document and ask me something relevant!",
            "sources_used": 0,
        }

    # ── Stage 1: Query Expansion (with pronoun resolution) ──────────────────
    expanded_queries = await llm_svc.expand_query(
        query.question,
        conversation_history=query.conversation_history,
    )
    resolved_query = expanded_queries[0]  # Pronoun-resolved version of the question
    logger.info(f"Query expansion | resolved='{resolved_query[:60]}' | variants={len(expanded_queries)}")

    # ── Semantic Cache Check ─────────────────────────────────────────────────
    # Embed just the resolved query (fast, 1 vector) and check the cache.
    # If we hit, we skip the entire pipeline and return in ~100ms.
    cache_embedding = (await embedding_svc.generate_embeddings([resolved_query]))[0]
    cached_answer = semantic_cache.get(cache_embedding, resolved_query)
    if cached_answer is not None:
        logger.info("Returning answer from semantic cache")
        return {
            "question":    query.question,
            "answer":      cached_answer,
            "sources_used": 0,
            "cache_hit":   True,
        }

    # ── Stage 2: Embed all expanded queries for Pinecone search ───────────────
    # We already have the resolved_query embedding, embed the rest and combine.
    rest_embeddings  = await embedding_svc.generate_embeddings(expanded_queries[1:])
    query_embeddings = [cache_embedding] + rest_embeddings

    # ── Self-RAG Retrieval Loop ────────────────────────────────────────────────
    MAX_RETRIES = 2
    active_question = query.question
    final_chunks = []

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Retrieval attempt {attempt}/{MAX_RETRIES}")

        # Stage 3: Bi-Encoder Search
        candidate_chunks, seen_texts = [], set()
        for embedding in query_embeddings:
            for chunk in pinecone_svc.search_vectors(embedding, top_k=10):
                if chunk not in seen_texts:
                    seen_texts.add(chunk)
                    candidate_chunks.append(chunk)

        logger.info(f"Candidates pooled | count={len(candidate_chunks)}")

        if not candidate_chunks:
            return {
                "question": query.question,
                "answer": "I don't have any documents in my database to answer this.",
                "sources_used": 0,
            }

        # Stage 4: Cross-Encoder Re-ranking
        final_chunks = reranker_svc.rerank_chunks(active_question, candidate_chunks, top_k=5)

        # Stage 5: Self-RAG Grade
        grade = await llm_svc.grade_context(active_question, final_chunks)
        logger.info(f"Context grade | grade={grade} | attempt={attempt}")

        if grade == "sufficient":
            break

        if attempt < MAX_RETRIES:
            active_question = await llm_svc.rewrite_query(active_question)
            logger.info(f"Query rewritten | new_query={active_question[:80]}")
            rewritten_embeddings = await embedding_svc.generate_embeddings([active_question])
            query_embeddings = rewritten_embeddings
        else:
            logger.warning("Max retries reached — answering with best available context.")

    # ── Stage 6: Generation ───────────────────────────────────────────────────
    final_answer = await llm_svc.generate_answer(
        question=query.question,
        context_chunks=final_chunks,
        conversation_history=query.conversation_history,
    )

    # ── Cache the result for future similar queries ────────────────────────────
    semantic_cache.set(cache_embedding, resolved_query, final_answer)

    logger.info(f"Search complete | sources={len(final_chunks)}")

    return {
        "question": query.question,
        "answer": final_answer,
        "sources_used": len(final_chunks),
    }


# ─── Shared RAG Pipeline Helper ───────────────────────────────────────────────
async def _run_rag_pipeline(question: str) -> tuple[str, list[str]]:
    """Runs the core RAG pipeline for one question. Used by both /search and /evaluate."""
    expanded_queries = await llm_svc.expand_query(question)
    query_embeddings = await embedding_svc.generate_embeddings(expanded_queries)

    candidate_chunks, seen_texts = [], set()
    for embedding in query_embeddings:
        for chunk in pinecone_svc.search_vectors(embedding, top_k=10):
            if chunk not in seen_texts:
                seen_texts.add(chunk)
                candidate_chunks.append(chunk)

    if not candidate_chunks:
        return "No relevant documents found in the database.", []

    final_chunks = reranker_svc.rerank_chunks(question, candidate_chunks, top_k=5)
    answer = await llm_svc.generate_answer(
        question=question, context_chunks=final_chunks, conversation_history=None
    )
    return answer, final_chunks


@app.post("/evaluate")
@limiter.limit("2/minute")  # Evaluation is expensive — strict cap
async def run_evaluation(request: Request, eval_req: EvaluationRequest):
    """
    Runs the full RAG pipeline on each test case, then evaluates
    the results with Ragas and returns structured metric scores.
    """
    if not eval_req.test_cases:
        error_response(400, "No test cases provided.")
    if len(eval_req.test_cases) > 15:
        error_response(400, "Maximum 15 test cases per evaluation run.")

    # Lazy-load all services
    global embedding_svc, pinecone_svc, llm_svc, reranker_svc
    if embedding_svc is None:
        embedding_svc = EmbeddingService()
    if pinecone_svc is None:
        pinecone_svc = PineconeService()
        pinecone_svc.ensure_index_exists()
    if reranker_svc is None:
        reranker_svc = RerankerService()
    if llm_svc is None:
        llm_svc = LLMService()

    logger.info(f"Evaluation started | test_cases={len(eval_req.test_cases)}")

    # Run the RAG pipeline for every test case
    questions, answers, contexts, ground_truths = [], [], [], []
    for tc in eval_req.test_cases:
        answer, chunks = await _run_rag_pipeline(tc.question)
        questions.append(tc.question)
        answers.append(answer)
        contexts.append(chunks)
        ground_truths.append(tc.ground_truth)

    # Run Ragas in a thread pool so we don't block the async event loop
    import asyncio
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    ragas_dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })
    ragas_llm        = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)
    ragas_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY)

    def _ragas_sync():
        from ragas import evaluate as ragas_eval
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        return ragas_eval(
            dataset=ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

    results = await asyncio.to_thread(_ragas_sync)
    df = results.to_pandas()

    def _safe_mean(col: str):
        return round(float(df[col].mean()), 3) if col in df.columns else None

    def _safe_val(row, col: str):
        v = row.get(col)
        try:
            f = float(v)
            return round(f, 3) if str(v) != "nan" else None
        except (TypeError, ValueError):
            return None

    aggregate = {
        "faithfulness":      _safe_mean("faithfulness"),
        "answer_relevancy":  _safe_mean("answer_relevancy"),
        "context_precision": _safe_mean("context_precision"),
        "context_recall":    _safe_mean("context_recall"),
    }

    per_question = []
    for i, (_, row) in enumerate(df.iterrows()):
        per_question.append({
            "question":          questions[i],
            "answer":            answers[i],
            "faithfulness":      _safe_val(row, "faithfulness"),
            "answer_relevancy":  _safe_val(row, "answer_relevancy"),
            "context_precision": _safe_val(row, "context_precision"),
            "context_recall":    _safe_val(row, "context_recall"),
        })

    logger.info(f"Evaluation complete | aggregate={aggregate}")
    return {"aggregate": aggregate, "per_question": per_question}
