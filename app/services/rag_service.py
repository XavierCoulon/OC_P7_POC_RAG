"""RAG service for managing the vector index and retrieval chain."""

import logging
import os
import time
from typing import Optional, Dict, Any
from datetime import datetime

from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import SecretStr

from app.core.config import settings
from app.core.index_manager import IndexManager
from app.core.embeddings import EmbeddingProvider, create_embedding_provider
from app.core.prompts import get_rag_prompt, get_chat_response
from app.core.classification import classify_query_intent, INTENT_CHAT
from app.external.openagenda_fetch import fetch_all_events, BASE_URL
from app.utils.document_converter import DocumentBuilder

logger = logging.getLogger(__name__)


class RAGService:
    """Service for managing RAG pipeline, vector store, and retrieval chain."""

    # Constants for RAG configuration
    RETRIEVER_K = 6  # Number of documents to retrieve
    DISTANCE_STRATEGY = "COSINE"
    SUPPORTED_PROVIDERS = ["mistral", "huggingface"]

    def __init__(self):
        """Initialize RAG service."""
        # Storage for multiple embedding providers and their indices
        self.embedding_providers: Dict[str, EmbeddingProvider] = {}
        self.vector_stores: Dict[str, Optional[FAISS]] = {}
        self.rag_chains: Dict[
            str, Optional[Runnable[Dict[str, Any], Dict[str, Any]]]
        ] = {}
        self.index_managers: Dict[str, Optional[IndexManager]] = {}

        # Initialize storage for each provider
        for provider in self.SUPPORTED_PROVIDERS:
            self.vector_stores[provider] = None
            self.rag_chains[provider] = None
            self.index_managers[provider] = None

        self.llm: Optional[ChatMistralAI] = None

    def _get_or_create_embedding_provider(
        self, provider_name: str
    ) -> EmbeddingProvider:
        """Get or create an embedding provider instance.

        Args:
            provider_name: Name of the provider ("mistral" or "huggingface")

        Returns:
            EmbeddingProvider instance

        Raises:
            ValueError: If provider_name is not supported
        """
        if provider_name not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported embedding provider: {provider_name}. "
                f"Choose from: {', '.join(self.SUPPORTED_PROVIDERS)}"
            )

        if provider_name not in self.embedding_providers:
            if provider_name == "mistral":
                self.embedding_providers[provider_name] = create_embedding_provider(
                    "mistral", api_key=settings.mistral_api_key
                )
            elif provider_name == "huggingface":
                self.embedding_providers[provider_name] = create_embedding_provider(
                    "huggingface", model_name=settings.huggingface_model_name
                )

        return self.embedding_providers[provider_name]

    def _initialize_llm(self) -> ChatMistralAI:
        """Initialize Mistral LLM.

        Returns:
            ChatMistralAI instance
        """
        return ChatMistralAI(
            api_key=SecretStr(settings.mistral_api_key),
            model_name="mistral-small-latest",
            temperature=0.3,
        )

    def _get_llm(self) -> ChatMistralAI:
        """Get or create LLM instance (lazy initialization).

        Returns:
            Cached ChatMistralAI instance
        """
        if self.llm is None:
            self.llm = self._initialize_llm()
        return self.llm

    def _invoke_with_retry(
        self,
        invokable,
        input_data: Dict[str, Any],
        max_retries: int = 3,
        initial_delay: int = 1,
    ) -> Dict[str, Any]:
        """Invoke a runnable with retry logic for 429 errors.

        Uses exponential backoff strategy to handle rate limits gracefully.

        Args:
            invokable: The runnable to invoke (rag_chain or llm)
            input_data: Input dictionary for the runnable
            max_retries: Maximum number of retry attempts (default: 3)
            initial_delay: Initial delay in seconds (default: 1)

        Returns:
            Result from the invocation or error dict
        """
        for attempt in range(max_retries):
            try:
                logger.debug(f"Invoke attempt {attempt + 1}/{max_retries}")
                result = invokable.invoke(input_data)
                return result

            except Exception as e:
                error_msg = str(e)

                # Check for rate limit error
                if (
                    "429" in error_msg
                    or "capacity exceeded" in error_msg
                    or "rate limit" in error_msg.lower()
                ):
                    if attempt < max_retries - 1:
                        wait_time = initial_delay * (2**attempt)  # Exponential backoff
                        logger.warning(
                            f"Rate limit error (429). Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit error after {max_retries} attempts")
                        return {
                            "answer": "Le service Mistral est surchargé. Veuillez réessayer dans quelques minutes.",
                            "error": "429",
                        }
                else:
                    # Other error - don't retry
                    logger.error(f"Error during invocation: {error_msg}")
                    raise

        return {"answer": "Erreur inconnue", "error": "unknown"}

    def _create_rag_chain(
        self, vector_store: FAISS
    ) -> Runnable[Dict[str, Any], Dict[str, Any]]:
        """Create RAG chain from vector store.

        Args:
            vector_store: FAISS vector store

        Returns:
            RAG chain instance (Runnable that accepts dict with 'input' key)
        """
        llm = self._get_llm()

        # Get RAG prompt with location_department from environment
        rag_prompt = get_rag_prompt()
        prompt = ChatPromptTemplate.from_template(rag_prompt)
        stuff_chain = create_stuff_documents_chain(llm, prompt)

        rag_chain = create_retrieval_chain(
            retriever=vector_store.as_retriever(search_kwargs={"k": self.RETRIEVER_K}),
            combine_docs_chain=stuff_chain,
        )

        return rag_chain

    def rebuild_index(self, provider: str = "mistral") -> Dict[str, Any]:
        """Rebuild the RAG index from OpenAgenda API for a specific provider.

        Args:
            provider: Embedding provider to rebuild for ("mistral" or "huggingface")

        Returns:
            Dictionary with rebuild status and metadata
        """
        try:
            logger.info(f"Starting index rebuild for provider: {provider}...")

            # Validate provider
            if provider not in self.SUPPORTED_PROVIDERS:
                return {
                    "status": "error",
                    "message": f"Unsupported provider: {provider}. Choose from: {', '.join(self.SUPPORTED_PROVIDERS)}",
                }

            # Fetch events
            logger.info("Fetching events from OpenAgenda...")
            events = fetch_all_events(BASE_URL)
            logger.info(f"✓ Fetched {len(events)} events")

            # Convert to documents (each event becomes multiple chunked documents)
            logger.info("Converting events to documents and chunking...")
            builder = DocumentBuilder()
            all_documents = [doc for event in events for doc in builder.build(event)]
            logger.info(f"✓ Created {len(all_documents)} chunks")

            # Get embedding provider
            logger.info(f"Getting embedding provider: {provider}...")
            embedding_provider = self._get_or_create_embedding_provider(provider)
            embeddings = embedding_provider.get_embeddings()
            distance_strategy = embedding_provider.get_distance_strategy()
            logger.info(
                f"✓ Using {provider} embeddings with {distance_strategy} distance"
            )

            # Create embeddings and vector store
            logger.info("Creating embeddings and vector store...")
            assert embeddings is not None, "Embeddings initialization failed"
            vector_store = FAISS.from_documents(
                documents=all_documents,
                embedding=embeddings,
                distance_strategy=distance_strategy,
            )
            logger.info(
                f"✓ Created FAISS index with {vector_store.index.ntotal} vectors"
            )

            # Save index
            logger.info("Saving index to disk...")
            index_manager = IndexManager(str(settings.index_dir / provider))
            metadata = {
                "provider": provider,
                "total_events": len(events),
                "total_chunks": len(all_documents),
                "total_vectors": vector_store.index.ntotal,
                "distance_strategy": distance_strategy,
                "rebuilt_at": datetime.now().isoformat(),
            }
            index_manager.save_index(vector_store, metadata)

            # Create RAG chain
            logger.info("Creating RAG chain...")
            rag_chain = self._create_rag_chain(vector_store)

            # Update instance state
            self.vector_stores[provider] = vector_store
            self.rag_chains[provider] = rag_chain
            self.index_managers[provider] = index_manager

            logger.info(f"✓ Index rebuild for {provider} completed successfully")
            return {
                "status": "success",
                "provider": provider,
                "message": f"Index rebuilt successfully for {provider}",
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"✗ Index rebuild for {provider} failed: {e}")
            return {
                "status": "error",
                "provider": provider,
                "message": str(e),
            }

    def load_index(self, provider: str = "mistral") -> Dict[str, Any]:
        """Load RAG index from disk for a specific provider.

        Args:
            provider: Embedding provider to load index for ("mistral" or "huggingface")

        Returns:
            Dictionary with load status and metadata
        """
        try:
            logger.info(f"Loading index from disk for provider: {provider}...")

            # Validate provider
            if provider not in self.SUPPORTED_PROVIDERS:
                return {
                    "status": "error",
                    "message": f"Unsupported provider: {provider}. Choose from: {', '.join(self.SUPPORTED_PROVIDERS)}",
                }

            # Get embedding provider
            embedding_provider = self._get_or_create_embedding_provider(provider)
            embeddings = embedding_provider.get_embeddings()
            assert embeddings is not None, "Embeddings initialization failed"

            # Try to load index
            index_manager = IndexManager(str(settings.index_dir / provider))
            vector_store = index_manager.load_index(embeddings)

            if vector_store is None:
                logger.warning(f"No index found for {provider}, skipping load")
                return {
                    "status": "not_found",
                    "provider": provider,
                    "message": f"No index found for {provider}. Call /rebuild first.",
                }

            # Create RAG chain
            rag_chain = self._create_rag_chain(vector_store)

            # Update instance state
            self.vector_stores[provider] = vector_store
            self.rag_chains[provider] = rag_chain
            self.index_managers[provider] = index_manager

            metadata = index_manager.get_index_info()
            logger.info(f"✓ Index loaded successfully for {provider}")

            return {
                "status": "success",
                "provider": provider,
                "message": f"Index loaded successfully for {provider}",
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"✗ Index load for {provider} failed: {e}")
            return {
                "status": "error",
                "provider": provider,
                "message": str(e),
            }

    def get_rag_chain(
        self, provider: str = "mistral"
    ) -> Optional[Runnable[Dict[str, Any], Dict[str, Any]]]:
        """Get the current RAG chain for a provider.

        Args:
            provider: Embedding provider name

        Returns:
            RAG chain or None if not initialized
        """
        return self.rag_chains.get(provider)

    def get_index_info(self, provider: str = "mistral") -> Dict[str, Any]:
        """Get information about the current index for a provider.

        Args:
            provider: Embedding provider name

        Returns:
            Index information dictionary
        """
        index_manager = self.index_managers.get(provider)

        if index_manager is None:
            return {
                "status": "not_initialized",
                "provider": provider,
            }

        return index_manager.get_index_info()

    def is_ready(self, provider: str = "mistral") -> bool:
        """Check if RAG service is ready for queries with a specific provider.

        Args:
            provider: Embedding provider name

        Returns:
            True if rag_chain is initialized, False otherwise
        """
        return self.rag_chains.get(provider) is not None

    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all embedding providers.

        Returns:
            Dictionary with provider status information
        """
        result = {}
        for provider in self.SUPPORTED_PROVIDERS:
            result[provider] = {
                "available": self.is_ready(provider),
                "info": self.get_index_info(provider),
            }
        return result

    def classify_intent(self, query: str) -> str:
        """Classify the intent of a user query.

        Args:
            query: User's question

        Returns:
            Intent classification: "RAG" or "CHAT"
        """
        llm = self._get_llm()
        return classify_query_intent(query, llm)

    def _extract_events_from_context(self, context_docs: list) -> list:
        """Extract event information from context documents.

        Args:
            context_docs: List of LangChain Document objects from RAG context

        Returns:
            List of dictionaries with event information
        """
        events = []
        seen_titles = set()  # Avoid duplicates

        for doc in context_docs:
            metadata = doc.metadata or {}
            title = metadata.get("originagenda_title", "Événement sans titre")

            # Avoid duplicate titles in results
            if title not in seen_titles:
                seen_titles.add(title)
                events.append(
                    {
                        "title": title,
                        "location": f"{metadata.get('location_city', 'Lieu non spécifié')} ({metadata.get('location_postalcode', 'CP')})",
                        "start_date": metadata.get(
                            "firstdate_begin", "Date non spécifiée"
                        ),
                        "url": metadata.get("canonicalurl", None),
                    }
                )

        return events

    def answer_question(
        self, question: str, provider: str = "mistral"
    ) -> Dict[str, Any]:
        """Answer a question using RAG or return a chat response.

        Args:
            question: User's question
            provider: Embedding provider to use for RAG ("mistral" or "huggingface")

        Returns:
            Dictionary with question, answer, intent, and provider used
        """
        try:
            # Validate provider
            if provider not in self.SUPPORTED_PROVIDERS:
                return {
                    "status": "error",
                    "question": question,
                    "provider": provider,
                    "answer": f"Unsupported provider: {provider}",
                    "intent": None,
                }

            # Classify intent with retry logic
            intent = self.classify_intent(question)

            events = []  # Events used for RAG response

            if intent == INTENT_CHAT:
                # Return a friendly chat response
                logger.info(f"Chat intent detected for: '{question[:50]}...'")
                answer = get_chat_response()
            else:  # INTENT_RAG
                # Use RAG chain to answer with retry logic
                logger.info(
                    f"RAG intent detected for: '{question[:50]}...' using {provider}"
                )
                if not self.is_ready(provider):
                    answer = f"Je n'ai pas accès à l'index d'événements pour le provider {provider}. Veuillez rebuilder l'index."
                else:
                    rag_chain = self.get_rag_chain(provider)
                    # Type: is_ready() ensures rag_chain is not None
                    result = self._invoke_with_retry(
                        rag_chain,  # type: ignore
                        {"input": question},
                        max_retries=3,
                        initial_delay=1,
                    )

                    if "error" in result:
                        answer = result["answer"]
                    else:
                        answer = result.get("answer", "Aucune réponse générée")
                        # Check if model declined to answer (should not return events)
                        # Case 1: Geographic validation triggered
                        # Case 2: Truly no events found (not just "no events of that style")
                        if "Je suis spécialisé uniquement dans" in answer:
                            logger.info(
                                "Geographic validation triggered - no events returned"
                            )
                            events = []
                        elif (
                            "Aucun événement correspondant trouvé" in answer
                            and "de ce style" not in answer
                        ):
                            # Only block if truly no events, not if saying "no concerts but here are other events"
                            logger.info("No events found - no events returned")
                            events = []
                        else:
                            # Extract context documents (source events)
                            context = result.get("context", [])
                            events = self._extract_events_from_context(context)
                            logger.info(
                                f"✓ Extracted {len(events)} source events from context"
                            )

            return {
                "status": "success",
                "question": question,
                "answer": answer,
                "events": events,
                "intent": intent,
                "provider": provider,
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "status": "error",
                "question": question,
                "answer": f"Erreur lors du traitement: {str(e)}",
                "intent": None,
                "provider": provider,
                "events": [],
            }
