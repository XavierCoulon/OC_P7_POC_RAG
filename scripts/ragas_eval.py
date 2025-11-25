#!/usr/bin/env python3
"""Ragas evaluation - Test RAG quality using the RAGService directly with Mistral LLM."""

import argparse
import os
import sys
from typing import Dict, List

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

# Import RAG service
from app.services.rag_service import RAGService

# Load .env file
# Add parent directory to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


def query_rag_service(rag_service: RAGService, question: str, provider: str) -> Dict:
    """Query RAG service directly.

    Args:
        rag_service: RAGService instance
        question: The question to ask
        provider: Embedding provider ("mistral" or "huggingface")

    Returns:
        RAG response with answer and context
    """
    return rag_service.answer_question(question, provider=provider)


def eval_rag_service(
    test_questions: List[str],
    ground_truths: List[str],
    provider: str = "mistral",
):
    """Evaluate RAG by querying RAGService directly and scoring with Mistral.

    Args:
        test_questions: Questions to test
        ground_truths: Expected answers for comparison
        provider: Embedding provider ("mistral" or "huggingface")
    """
    # Initialize RAG service
    print(f"🔧 Initializing RAG service with {provider} provider...\n")
    rag_service = RAGService()

    # Load index
    print(f"📚 Loading index for {provider}...")
    load_result = rag_service.load_index(provider=provider)
    if load_result["status"] != "success":
        print(f"❌ Failed to load index: {load_result['message']}")
        return

    print(f"✓ Index loaded: {load_result['metadata'].get('total_vectors')} vectors\n")

    # Query RAG service
    print(f"🔄 Querying RAG service ({provider})...\n")

    questions = []
    contexts_list = []
    answers = []

    for q in test_questions:
        try:
            result = query_rag_service(rag_service, q, provider=provider)
            if result["status"] != "success":
                print(f"✗ Error: {result['answer']}")
                continue

            # Only evaluate RAG responses with context
            # Skip CHAT intent and responses without context (geographic validation, errors)
            intent = result.get("intent", "rag")
            context_docs = result.get("context", [])

            if intent == "chat":
                print(f"⊘ {q[:50]}... (CHAT intent - skipped for Ragas)")
                continue

            if not context_docs:
                print(f"⊘ {q[:50]}... (no context - skipped for Ragas)")
                continue

            # This is a RAG response with context - include in evaluation
            questions.append(q)
            context_strings = [doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in context_docs]
            contexts_list.append(context_strings)
            answers.append(result.get("answer", ""))
            print(f"✓ {q[:50]}... (RAG intent, {len(context_strings)} contexts)")

        except Exception as e:
            print(f"✗ Error querying: {e}")
            continue

    if not questions:
        print("❌ No RAG responses with context found. Cannot evaluate with Ragas.")
        return

    # Create Ragas dataset
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "contexts": contexts_list,
            "answer": answers,
            "ground_truth": ground_truths[: len(questions)],
        }
    )

    print(f"\n📊 Evaluating {len(questions)} responses with Ragas using Mistral...\n")

    try:
        # Get Mistral LLM for evaluation (Mistral is the generation LLM, independent of embedding provider)
        mistral_llm = rag_service._get_llm()

        # Get embeddings from RAGService to avoid OpenAI default
        embedding_provider = rag_service._get_or_create_embedding_provider(provider)
        embeddings = embedding_provider.get_embeddings()

        # Run evaluation - pass the ChatMistralAI directly as LLM
        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=mistral_llm,
            embeddings=embeddings,
        )

        # Convert results to dictionary
        results_dict = results.to_pandas().to_dict("list")

        # Convert lists to means (Ragas returns list of scores per sample)
        import statistics

        faithfulness_score = (
            statistics.mean(results_dict["faithfulness"])
            if isinstance(results_dict["faithfulness"], list)
            else results_dict["faithfulness"]
        )
        answer_relevancy_score = (
            statistics.mean(results_dict["answer_relevancy"])
            if isinstance(results_dict["answer_relevancy"], list)
            else results_dict["answer_relevancy"]
        )
        context_recall_score = (
            statistics.mean(results_dict["context_recall"])
            if isinstance(results_dict["context_recall"], list)
            else results_dict["context_recall"]
        )
        context_precision_score = (
            statistics.mean(results_dict["context_precision"])
            if isinstance(results_dict["context_precision"], list)
            else results_dict["context_precision"]
        )

        print("\n✅ Results:")
        print(f"  Faithfulness:      {faithfulness_score:.1%}")
        print(f"  Answer Relevancy:  {answer_relevancy_score:.1%}")
        print(f"  Context Recall:    {context_recall_score:.1%}")
        print(f"  Context Precision: {context_precision_score:.1%}")

        overall = (faithfulness_score + answer_relevancy_score + context_recall_score + context_precision_score) / 4

        status = "✅" if overall >= 0.8 else "⚠️" if overall >= 0.7 else "❌"
        print(f"\n{status} Overall: {overall:.1%}")

    except Exception as e:
        print(f"❌ Ragas error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate RAG quality using Ragas with specified embedding provider")
    parser.add_argument(
        "--provider",
        choices=["mistral", "huggingface"],
        default="mistral",
        help="Embedding provider to use (default: mistral)",
    )
    args = parser.parse_args()

    # Test questions - Based on RAG context, geography constraints, and intent classification
    test_qs = [
        # RAG Intent: Specific events in target department (Pyrénées-Atlantiques)
        "Quels événements culturels sont proposés à Pau en 2025?",
        # RAG Intent: Category-specific search within department
        "Y a-t-il des concerts ou festivals de musique à Bayonne?",
        # RAG Intent: Date-specific search
        "Quels événements pour enfants en juillet?",
    ]

    # Expected ground truths for evaluation
    ground_truths = [
        "Pau accueille des événements culturels diversifiés en 2025.",
        "Des festivals musicaux sont organisés dans la ville de Bayonne.",
        "Des activités pour enfants en juillet dans le département.",
    ]

    eval_rag_service(test_qs, ground_truths, provider=args.provider)
