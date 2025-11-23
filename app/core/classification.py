"""Query intent classification logic."""

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from app.core.prompts import CLASSIFICATION_PROMPT

# Constants for possible intents
INTENT_RAG = "RAG"
INTENT_CHAT = "CHAT"
DEFAULT_INTENT = INTENT_RAG  # Default to RAG to prioritize event search


def classify_query_intent(query: str, llm: ChatMistralAI) -> str:
    """Classify the intent of a user query.

    Args:
        query: The user's question.
        llm: Initialized ChatMistralAI instance.

    Returns:
        The detected intent ("RAG" or "CHAT").
    """
    prompt = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)

    try:
        logging.info(f"Classifying query: '{query[:50]}...'")

        # Invoke the LLM with the prompt
        response = llm.invoke(prompt.format(question=query))
        intent = str(response.content).strip().upper()

        if intent == INTENT_RAG:
            logging.info(f"Intent detected: {INTENT_RAG}")
            return INTENT_RAG
        elif intent == INTENT_CHAT:
            logging.info(f"Intent detected: {INTENT_CHAT}")
            return INTENT_CHAT
        else:
            logging.warning(f"Unclear classification received: '{intent}'. Using default intent: {DEFAULT_INTENT}")
            return DEFAULT_INTENT

    except Exception as e:
        logging.error(f"Error classifying query: {e}")
        return DEFAULT_INTENT
