"""RAG service for managing the vector index and retrieval chain."""

import logging
import os
import time
from typing import Optional, Dict, Any
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import SecretStr

from app.core.config import settings
from app.core.index_manager import IndexManager
from app.core.prompts import get_rag_prompt, get_chat_response
from app.core.classification import classify_query_intent, INTENT_CHAT
from app.external.openagenda_fetch import fetch_all_events, BASE_URL
from app.utils.document_converter import event_to_langchain_document

logger = logging.getLogger(__name__)


class RAGService:
    """Service for managing RAG pipeline, vector store, and retrieval chain."""

    def __init__(self):
        """Initialize RAG service."""
        self.rag_chain: Optional[Runnable[Dict[str, Any], Dict[str, Any]]] = None
        self.index_manager: Optional[IndexManager] = None
        self.vector_store: Optional[FAISS] = None
        self.llm: Optional[ChatMistralAI] = None

    def _initialize_embeddings(self) -> MistralAIEmbeddings:
        """Initialize Mistral embeddings.

        Returns:
            MistralAIEmbeddings instance
        """
        return MistralAIEmbeddings(
            model="mistral-embed", api_key=SecretStr(settings.mistral_api_key)
        )

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
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain=stuff_chain,
        )

        return rag_chain

    def rebuild_index(self) -> Dict[str, Any]:
        """Rebuild the RAG index from OpenAgenda API.

        Returns:
            Dictionary with rebuild status and metadata
        """
        try:
            logger.info("Starting index rebuild...")

            # Fetch events
            logger.info("Fetching events from OpenAgenda...")
            events = fetch_all_events(BASE_URL)
            logger.info(f"✓ Fetched {len(events)} events")

            # Convert to documents
            logger.info("Converting events to documents...")
            documents = [event_to_langchain_document(event) for event in events]
            logger.info(f"✓ Created {len(documents)} documents")

            # Split documents
            logger.info("Chunking documents...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " ", ""],
            )
            split_documents = splitter.split_documents(documents)
            logger.info(f"✓ Created {len(split_documents)} chunks")

            # Create embeddings and vector store
            logger.info("Creating embeddings and vector store...")
            embeddings = self._initialize_embeddings()
            vector_store = FAISS.from_documents(
                documents=split_documents,
                embedding=embeddings,
                distance_strategy="COSINE",
            )
            logger.info(
                f"✓ Created FAISS index with {vector_store.index.ntotal} vectors"
            )

            # Save index
            logger.info("Saving index to disk...")
            index_manager = IndexManager(str(settings.index_dir))
            metadata = {
                "total_events": len(events),
                "total_chunks": len(split_documents),
                "total_vectors": vector_store.index.ntotal,
                "distance_strategy": "COSINE",
                "rebuilt_at": datetime.now().isoformat(),
            }
            index_manager.save_index(vector_store, metadata)

            # Create RAG chain
            logger.info("Creating RAG chain...")
            rag_chain = self._create_rag_chain(vector_store)

            # Update instance state
            self.vector_store = vector_store
            self.rag_chain = rag_chain
            self.index_manager = index_manager

            logger.info("✓ Index rebuild completed successfully")
            return {
                "status": "success",
                "message": "Index rebuilt successfully",
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"✗ Index rebuild failed: {e}")
            return {"status": "error", "message": str(e)}

    def load_index(self) -> Dict[str, Any]:
        """Load RAG index from disk.

        Returns:
            Dictionary with load status and metadata
        """
        try:
            logger.info("Loading index from disk...")

            embeddings = self._initialize_embeddings()
            index_manager = IndexManager(str(settings.index_dir))

            # Try to load index
            vector_store = index_manager.load_index(embeddings)

            if vector_store is None:
                logger.warning("No index found, skipping load")
                return {
                    "status": "not_found",
                    "message": "No index found on disk. Call /rebuild first.",
                }

            # Create RAG chain
            rag_chain = self._create_rag_chain(vector_store)

            # Update instance state
            self.vector_store = vector_store
            self.rag_chain = rag_chain
            self.index_manager = index_manager

            metadata = index_manager.get_index_info()
            logger.info("✓ Index loaded successfully")

            return {
                "status": "success",
                "message": "Index loaded successfully",
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"✗ Index load failed: {e}")
            return {"status": "error", "message": str(e)}

    def get_rag_chain(self) -> Optional[Runnable[Dict[str, Any], Dict[str, Any]]]:
        """Get the current RAG chain.

        Returns:
            RAG chain or None if not initialized
        """
        return self.rag_chain

    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index.

        Returns:
            Index information dictionary
        """
        if self.index_manager is None:
            return {"status": "not_initialized"}

        return self.index_manager.get_index_info()

    def is_ready(self) -> bool:
        """Check if RAG service is ready for queries.

        Returns:
            True if rag_chain is initialized, False otherwise
        """
        return self.rag_chain is not None

    def classify_intent(self, query: str) -> str:
        """Classify the intent of a user query.

        Args:
            query: User's question

        Returns:
            Intent classification: "RAG" or "CHAT"
        """
        llm = self._get_llm()
        return classify_query_intent(query, llm)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using RAG or return a chat response.

        Args:
            question: User's question

        Returns:
            Dictionary with question, answer, and intent
        """
        try:
            # Classify intent with retry logic
            intent = self.classify_intent(question)

            if intent == INTENT_CHAT:
                # Return a friendly chat response
                logger.info(f"Chat intent detected for: '{question[:50]}...'")
                answer = get_chat_response()
            else:  # INTENT_RAG
                # Use RAG chain to answer with retry logic
                logger.info(f"RAG intent detected for: '{question[:50]}...'")
                if not self.is_ready():
                    answer = "Je n'ai pas accès à l'index d'événements. Veuillez relancer l'application ou rebuilder l'index."
                else:
                    # Type: is_ready() ensures rag_chain is not None
                    result = self._invoke_with_retry(
                        self.rag_chain,  # type: ignore
                        {"input": question},
                        max_retries=3,
                        initial_delay=1,
                    )
                    if "error" in result:
                        answer = result["answer"]
                    else:
                        answer = result.get("answer", "Aucune réponse générée")

            return {
                "status": "success",
                "question": question,
                "answer": answer,
                "intent": intent,
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "status": "error",
                "question": question,
                "answer": f"Erreur lors du traitement: {str(e)}",
                "intent": None,
            }
