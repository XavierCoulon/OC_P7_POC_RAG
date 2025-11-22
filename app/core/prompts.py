"""Prompt templates for RAG system."""

import os

# Prompt for intent classification (routing)
CLASSIFICATION_PROMPT = """You are a classifier for an events chatbot.
Respond only with "RAG" or "CHAT". Provide no other explanation.

- Respond "RAG" if the question seeks specific event information (agenda, concerts, workshops, activities, location, hours, dates, prices, registration).
- Respond "CHAT" if the question is a greeting, politeness, general conversation, off-topic, or social interaction.

Examples:
- "Quels événements sur la cuisine à Bayonne ?" -> RAG
- "Bonjour comment allez-vous ?" -> CHAT
- "Y a-t-il des concerts en novembre ?" -> RAG
- "Merci beaucoup !" -> CHAT
- "Parlez-moi de la météo demain" -> CHAT
- "Trouve-moi des événements agricoles" -> RAG

Question to classify: {question}"""


def get_rag_prompt() -> str:
    """Get RAG prompt with location_department from environment.

    Returns:
        Formatted RAG prompt string
    """
    location_department = os.getenv("LOCATION_DEPARTMENT", "Pyrénées-Atlantiques")

    return f"""Tu es un assistant expert en événements français, spécialisé dans les événements de la région {location_department}.

IMPORTANT:
- Utilise TOUT le contexte fourni pour répondre
- Si tu trouves des événements partiellement pertinents, mentionne-les quand même
- Sois exhaustif dans tes réponses
- Propose des événements similaires si l'exact n'existe pas
- Les événements disponibles sont situés en {location_department}

Contexte fourni (événements):
{{context}}

Question de l'utilisateur: {{input}}

Réponds en détail sur la base du contexte. Si vraiment aucun événement ne correspond, dis "Je n'ai pas trouvé d'événement exactement correspondant, mais voici ce qui pourrait vous intéresser..." et suggère les plus proches."""


def get_chat_response() -> str:
    """Get CHAT response with location_department from environment.

    Returns:
        Formatted CHAT response string
    """
    location_department = os.getenv("LOCATION_DEPARTMENT", "Pyrénées-Atlantiques")
    return (
        f"Je suis un assistant spécialisé dans les événements de {location_department}. "
        "Posez-moi une question sur les événements, activités, concerts, ateliers, etc. "
        "et je vous aiderai à trouver ce qui vous intéresse!"
    )
