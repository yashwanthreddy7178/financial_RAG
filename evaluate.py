"""
evaluate.py — Ragas Evaluation Script for the Financial RAG Pipeline

Run this script SEPARATELY from the server (the server does NOT need to be running):
    python evaluate.py

This script will:
  1. Run your full RAG pipeline on a hand-crafted test dataset
  2. Collect the answers AND the retrieved context chunks
  3. Feed everything to Ragas for scientific scoring
  4. Print a results table with 4 metrics for each question

IMPORTANT: Make sure your .env file has valid OPENAI_API_KEY and PINECONE_API_KEY.
           Also make sure you have already ingested your NVIDIA PDF before running this.
"""

import asyncio
from dotenv import load_dotenv

# Load environment variables from .env BEFORE importing our services
load_dotenv()

from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService
from services.reranker_service import RerankerService
from services.llm_service import LLMService

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: THE TEST DATASET
# These are questions based on the NVIDIA 2025 Annual Report.
# 'ground_truth' is the correct answer you know from reading the document.
# ─────────────────────────────────────────────────────────────────────────────
TEST_DATASET = [
    {
        "question": "What was NVIDIA's total revenue for fiscal year 2025?",
        "ground_truth": "NVIDIA's total revenue for fiscal year 2025 was $130.5 billion."
    },
    {
        "question": "What is the name of NVIDIA's latest GPU architecture?",
        "ground_truth": "NVIDIA's latest GPU architecture is called Blackwell."
    },
    {
        "question": "What are the primary risk factors NVIDIA mentions regarding international sales?",
        "ground_truth": "NVIDIA cites multiple international risk factors including export control restrictions, economic and political conditions in countries where they operate, different legal standards for IP protection and employment practices, currency fluctuations, trade restrictions such as tariffs and sanctions, government lockdowns due to health issues, and risks from natural disasters, acts of war, or terrorism."
    },
    {
        "question": "How much did NVIDIA's Data Center segment revenue grow year over year?",
        "ground_truth": "NVIDIA's Data Center segment revenue grew approximately 142% year over year in fiscal 2025."
    },
    {
        "question": "What is NVIDIA's primary manufacturing model and who is their key partner?",
        "ground_truth": "NVIDIA uses a fabless manufacturing model, outsourcing all manufacturing to third-party partners. Their primary partners include TSMC (Taiwan Semiconductor Manufacturing Company) for semiconductor wafer fabrication, and Samsung Electronics as an additional manufacturing partner for wafer production."
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: THE PIPELINE RUNNER
# This runs our full RAG pipeline for a single question and returns both
# the final answer AND the list of context chunks (required by Ragas).
# ─────────────────────────────────────────────────────────────────────────────
async def run_pipeline_for_question(
    question: str,
    embedding_svc: EmbeddingService,
    pinecone_svc: PineconeService,
    reranker_svc: RerankerService,
    llm_svc: LLMService,
) -> tuple[str, list[str]]:
    """
    Runs the full RAG pipeline for one question.
    Returns: (answer_string, list_of_context_chunk_strings)
    """
    print(f"\n  → Testing: '{question[:60]}...'")

    # 1. Expand the query
    expanded_queries = await llm_svc.expand_query(question)

    # 2. Embed all variations
    query_embeddings = await embedding_svc.generate_embeddings(expanded_queries)

    # 3. Search Pinecone and pool results
    candidate_chunks = []
    seen_texts = set()
    for embedding in query_embeddings:
        chunks = pinecone_svc.search_vectors(embedding, top_k=10)
        for chunk in chunks:
            if chunk not in seen_texts:
                seen_texts.add(chunk)
                candidate_chunks.append(chunk)

    if not candidate_chunks:
        return "No relevant documents found.", []

    # 4. Re-rank and select top 5
    final_chunks = reranker_svc.rerank_chunks(question, candidate_chunks, top_k=5)

    # 5. Generate the answer
    answer = await llm_svc.generate_answer(
        question=question,
        context_chunks=final_chunks,
        conversation_history=None
    )

    return answer, final_chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: THE MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("  Financial RAG — Ragas Evaluation")
    print("=" * 60)

    # Initialize all services
    print("\n[1/3] Initializing services...")
    embedding_svc = EmbeddingService()
    pinecone_svc = PineconeService()
    pinecone_svc.ensure_index_exists()
    reranker_svc = RerankerService()
    llm_svc = LLMService()
    print("  ✅ All services ready.")

    # Run the pipeline for every test question
    print(f"\n[2/3] Running pipeline on {len(TEST_DATASET)} test questions...")
    questions, answers, contexts, ground_truths = [], [], [], []

    for test_case in TEST_DATASET:
        answer, retrieved_chunks = await run_pipeline_for_question(
            question=test_case["question"],
            embedding_svc=embedding_svc,
            pinecone_svc=pinecone_svc,
            reranker_svc=reranker_svc,
            llm_svc=llm_svc,
        )
        questions.append(test_case["question"])
        answers.append(answer)
        contexts.append(retrieved_chunks)       # Ragas expects a list of lists
        ground_truths.append(test_case["ground_truth"])

    print("\n  ✅ All pipeline runs complete.")

    # ── Run Ragas ─────────────────────────────────────────────────────────────
    print("\n[3/3] Running Ragas evaluation (this may take 1-2 minutes)...")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # Package our collected data into a Ragas-compatible HuggingFace Dataset
    ragas_dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    # Ragas uses LangChain wrappers for its own internal LLM calls
    ragas_llm = ChatOpenAI(model="gpt-4o-mini")
    ragas_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    results = evaluate(
        dataset=ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # ── Print the Report ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 RAGAS EVALUATION RESULTS")
    print("=" * 60)

    # Convert to a Pandas DataFrame for beautiful table printing
    df = results.to_pandas()

    # Show aggregate scores first
    print("\n🎯 AGGREGATE SCORES (average across all questions):")
    print(f"   Faithfulness       : {df['faithfulness'].mean():.3f}  (hallucination prevention)")
    print(f"   Answer Relevancy   : {df['answer_relevancy'].mean():.3f}  (on-topic answers)")
    print(f"   Context Precision  : {df['context_precision'].mean():.3f}  (retrieval quality)")
    print(f"   Context Recall     : {df['context_recall'].mean():.3f}  (retrieval completeness)")

    # Show per-question breakdown
    print("\n📋 PER-QUESTION BREAKDOWN:")
    # Newer Ragas versions rename 'question' to 'user_input'
    q_col = "user_input" if "user_input" in df.columns else "question"
    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if c in df.columns]
    df_display = df[[q_col] + metric_cols].copy()
    df_display[q_col] = df_display[q_col].str[:55] + "..."
    df_display = df_display.rename(columns={q_col: "question"})
    print(df_display.to_string(index=False))

    print("\n" + "=" * 60)
    print("  Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
