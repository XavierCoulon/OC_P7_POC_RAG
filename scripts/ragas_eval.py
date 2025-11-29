#!/usr/bin/env python3
"""Ragas evaluation - Test RAG quality using the RAGService directly with Mistral LLM."""

import argparse
import os
import statistics
import sys
from typing import Dict, List

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

# Add parent directory to path FIRST (before any imports from app)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file BEFORE importing app modules
load_dotenv()

# Import RAG service (now safe since .env is loaded and sys.path is set)
from app.services.rag_service import RAGService  # noqa: E402


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

    # Test Dataset - Organized by Category
    # Structure: question | ground_truth | type
    # Types: geo_search, category_search, date_filter, date_relative,
    # specific_detail, metadata_retrieval, complex_filter, out_of_scope
    TEST_DATASET = [
        # --- CATÉGORIE 1 : RECHERCHE GÉOGRAPHIQUE SIMPLE ---
        {
            "question": "Quels événements culturels sont proposés à Pau en 2025?",
            "ground_truth": "Pau accueille plusieurs événements culturels diversifiés en 2025.",
            "type": "geo_search",
        },
        {
            "question": "Y a-t-il des concerts ou festivals de musique à Bayonne?",
            "ground_truth": ("Oui, Bayonne accueille régulièrement des concerts et festivals " "musicaux."),
            "type": "category_search",
        },
        {
            "question": "Y a-t-il des pièces de théâtre dans le Béarn ?",
            "ground_truth": "Oui, au moins deux pièces de théâtre dans le Béarn",
            "type": "geo_search",
        },
        # # --- CATÉGORIE 2 : CONTRAINTES TEMPORELLES ---
        {
            "question": "Quels événements sont prévus en septembre?",
            "ground_truth": (
                "En septembre, il y a une journée de la formation à Pau, une "
                "présentation découverte de l'orgue à Bayonne nommé 'Nouveaux "
                "vitraux pour l'église Saint-Etienne', des ateliers 'Se former, "
                "pourquoi pas vous', un atelier 'Comprendre la sécurité sur les "
                "chantiers' à Pau"
            ),
            "type": "date_filter",
        },
        {
            "question": "Que faire ce week-end dans le Pays Basque?",
            "ground_truth": (
                "Ce week-end, il y a des marchés artisanaux et des spectacles " "dans les villes principales."
            ),
            "type": "date_relative",
        },
        {
            "question": "Y a-t-il des événements prévus en mars 2025?",
            "ground_truth": (
                "En mars, deux événements prévus dans les Pyrénées-Atlantiques : "
                "'la restauration pourquoi pas vous ?' à Pau le 4 mars 2025, une "
                "réunion d'information pour découvrir les métiers de la restauration, "
                "'Les RDV de la tech et de l'industrie au féminin' à Pau le "
                "24 mars 2025, pour explorer les métiers de l'industrie et de la tech."
            ),
            "type": "date_filter",
        },
        # # --- CATÉGORIE 3 : RECHERCHE SPÉCIFIQUE (Détails) ---
        {
            "question": "Où se déroule l'atelier 'La restauration pourquoi pas vous ?'",
            "ground_truth": "L'atelier 'La restauration pourquoi pas vous ?' se déroule à Pau.",
            "type": "specific_detail",
        },
        {
            "question": "Avez-vous des informations sur les événements gratuits?",
            "ground_truth": "Oui, plusieurs événements gratuits sont proposés incluant des concerts en plein air.",
            "type": "metadata_retrieval",
        },
        # # --- CATÉGORIE 4 : FILTRES COMPLEXES (Multi-critères) ---
        {
            "question": "Je cherche un spectacle gratuit à Oloron-Sainte-Marie.",
            "ground_truth": "À Oloron, des spectacles gratuits sont organisés notamment en plein air.",
            "type": "complex_filter",
        },
        {
            "question": "Quels concerts de musique classique sont proposés en novembre?",
            "ground_truth": "En novembre, plusieurs concerts de musique classique sont organisés.",
            "type": "complex_filter",
        },
    ]

    # Extract test data from structured dataset
    test_questions = [item["question"] for item in TEST_DATASET]
    ground_truths = [item["ground_truth"] for item in TEST_DATASET]

    print("📊 Test Dataset Summary:")
    print(f"  Total: {len(TEST_DATASET)} questions")
    print(
        "  Categories: geo_search, category_search, date_filter, "
        "date_relative, specific_detail, metadata_retrieval, complex_filter, "
        "out_of_scope\n"
    )

    # Count by type
    type_counts = {}
    for item in TEST_DATASET:
        test_type = item["type"]
        type_counts[test_type] = type_counts.get(test_type, 0) + 1

    for test_type, count in sorted(type_counts.items()):
        print(f"  {test_type}: {count}")
    print()

    eval_rag_service(test_questions, ground_truths, provider=args.provider)
