import logging
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# Constantes pour les intentions possibles
INTENT_RAG = "RAG"
INTENT_CHAT = "CHAT"
DEFAULT_INTENT = INTENT_RAG  # Choisir RAG par défaut pour privilégier la recherche


def classify_query_intent(query: str, llm: ChatMistralAI) -> str:
    """
    Classifie l'intention de la requête utilisateur en utilisant l'API Mistral via LangChain.

    Args:
            query: La question posée par l'utilisateur.
            llm: Instance ChatMistralAI initialisée.

    Returns:
            L'intention détectée ("RAG" ou "CHAT").
    """
    classification_system_prompt = """
	Votre rôle est de classifier l'intention de la question de l'utilisateur pour un chatbot d'événements.
	Répondez uniquement par "RAG" ou "CHAT". Ne fournissez aucune autre explication.

	- Répondez "RAG" si la question cherche des informations spécifiques sur les événements (agenda, concerts, ateliers, activités, lieu, horaires, dates, tarifs, inscriptions).
	- Répondez "CHAT" si la question est une salutation, une formule de politesse, une conversation générale, une question hors sujet, ou une simple interaction sociale.

	Exemples:
	- "Quels événements sur la cuisine à Bayonne ?" -> RAG
	- "Bonjour comment allez-vous ?" -> CHAT
	- "Y a-t-il des concerts en novembre ?" -> RAG
	- "Merci beaucoup !" -> CHAT
	- "Parlez-moi de la météo demain" -> CHAT
	- "Trouve-moi des événements agricoles" -> RAG

	Question à classifier: {question}
	"""

    prompt = ChatPromptTemplate.from_template(classification_system_prompt)

    try:
        logging.info(f"Classification de la requête: '{query[:50]}...'")

        # Invoke le LLM avec le prompt
        response = llm.invoke(prompt.format(question=query))
        intent = str(response.content).strip().upper()

        if intent == INTENT_RAG:
            logging.info(f"Intention détectée: {INTENT_RAG}")
            return INTENT_RAG
        elif intent == INTENT_CHAT:
            logging.info(f"Intention détectée: {INTENT_CHAT}")
            return INTENT_CHAT
        else:
            logging.warning(
                f"Classification non claire reçue: '{intent}'. Utilisation de l'intention par défaut: {DEFAULT_INTENT}"
            )
            return DEFAULT_INTENT

    except Exception as e:
        logging.error(f"Erreur lors de la classification de la requête: {e}")
        return DEFAULT_INTENT
