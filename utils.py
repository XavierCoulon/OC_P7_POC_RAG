import re
import time
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openagenda_fetch import Event


def clean_html_content(html_content: str) -> str:
    """Cleans HTML content and returns plain text.

    Args:
            html_content (str): The HTML content to be cleaned.

    Returns:
            str: The cleaned plain text.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def normalize_whitespace(text: str) -> str:
    """Normalizes whitespace in a string by replacing multiple spaces with a single space.

    Args:
                    text (str): The input string.

    Returns:
                    str: The string with normalized whitespace.
    """
    return re.sub(r"\s+", " ", text).strip()


def build_document(event: Event) -> str:
    """Build a formatted document from an Event object.

    Args:
        event (Event): The event to build a document from.

    Returns:
        str: The formatted document as a string.
    """
    title = event.title_fr or ""
    desc = event.description_fr or ""
    longdesc = clean_html_content(event.longdescription_fr or "")
    keywords = ", ".join(event.keywords_fr or [])

    city = event.location_city or ""
    dept = event.location_department or ""
    region = event.location_region or ""
    address = event.location_address or ""

    date_range = event.daterange_fr or ""
    start = event.firstdate_begin or ""
    end = event.firstdate_end or ""

    origin = event.originagenda_title or ""

    text = f"""
    Titre : {title}
    Description : {desc}
    Description longue : {longdesc}
    Mots-clés : {keywords}
    Ville : {city}
    Adresse : {address}
    Département : {dept}
    Région : {region}
    Date : {date_range}
    Début : {start}
    Fin : {end}
    Agenda d'origine : {origin}
    """

    return normalize_whitespace(text)


def chunk_event_document(event: Event):
    doc = build_document(event)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_text(doc)


def event_to_langchain_document(event: Event) -> Document:
    """Convert an Event to a LangChain Document optimized for RAG.

    Optimization strategy:
    - page_content: Rich text with title, descriptions and keywords for semantic search
    - metadata: Structured data for filtering and display

    Args:
            event (Event): The event to convert.

    Returns:
            Document: LangChain Document with optimized content and metadata.
    """
    # PAGE CONTENT: Rich semantic content for embedding search
    # Focus on title, descriptions and keywords which carry semantic meaning
    title = event.title_fr or ""
    short_desc = event.description_fr or ""
    long_desc = clean_html_content(event.longdescription_fr or "")
    keywords = event.keywords_fr or []

    # Build comprehensive page_content for better semantic search
    page_content = f"""{title}

{short_desc}

{long_desc}

Catégories: {', '.join(keywords)}"""

    page_content = normalize_whitespace(page_content)

    # METADATA: Structured information for filtering and display
    metadata = {
        # Identifiers
        "uid": event.uid,
        "slug": event.slug,
        "canonicalurl": event.canonicalurl,
        # Location data (for geographic filtering)
        "location_city": event.location_city,
        "location_department": event.location_department,
        "location_region": event.location_region,
        "location_address": event.location_address,
        "location_postalcode": event.location_postalcode,
        # Temporal data (for date filtering)
        "firstdate_begin": (
            str(event.firstdate_begin) if event.firstdate_begin else None
        ),
        "firstdate_end": str(event.firstdate_end) if event.firstdate_end else None,
        "lastdate_begin": str(event.lastdate_begin) if event.lastdate_begin else None,
        "lastdate_end": str(event.lastdate_end) if event.lastdate_end else None,
        # Source info
        "originagenda_title": event.originagenda_title,
        "originagenda_uid": event.originagenda_uid,
        # Event metadata
        "age_min": event.age_min,
        "age_max": event.age_max,
    }

    return Document(page_content=page_content, metadata=metadata)


def embed_with_retry(embedding_model, texts, retries=5, delay=1.0):
    for attempt in range(retries):
        try:
            return embedding_model.embed_documents(texts)
        except Exception as e:
            if "429" in str(e):
                time.sleep(delay * (attempt + 1))
                continue
            raise e
    raise RuntimeError("Max retry reached for embeddings")


def invoke_rag_with_retry(
    rag_chain, query: str, max_retries: int = 3, initial_delay: int = 2
) -> dict:
    """
    Invoke RAG chain avec gestion des erreurs 429 (rate limit).

    Utilise une stratégie de backoff exponentiel pour réessayer en cas de rate limit.

    Args:
            rag_chain: La chaîne RAG à invoquer
            query (str): La question à poser
            max_retries (int): Nombre maximum de tentatives (défaut: 3)
            initial_delay (int): Délai initial d'attente en secondes (défaut: 2)

    Returns:
            dict: Le résultat de l'invocation ou un message d'erreur
    """
    import time

    for attempt in range(max_retries):
        try:
            print(f"🔄 Tentative {attempt + 1}/{max_retries}...")
            result = rag_chain.invoke({"input": query})
            return result

        except Exception as e:
            error_msg = str(e)

            # Check si c'est une erreur 429
            if "429" in error_msg or "capacity exceeded" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = initial_delay * (2**attempt)  # Backoff exponentiel
                    print(
                        f"⏳ Erreur 429 (rate limit). Attente {wait_time}s avant retry..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Erreur 429 après {max_retries} tentatives. Abandon.")
                    return {
                        "answer": "Le service Mistral est surchargé. Veuillez réessayer dans quelques minutes."
                    }
            else:
                # Autre erreur
                print(f"❌ Erreur: {error_msg}")
                return {"answer": f"Erreur: {error_msg}"}

    return {"answer": "Erreur inconnue"}


def invoke_with_intent_routing(
    query: str, llm, rag_chain, max_retries: int = 3, initial_delay: int = 2
) -> dict:
    """
    Invoque le chatbot avec routing intelligent basé sur l'intention de la requête.

    - Si RAG: utilise la chaîne RAG avec contexte
    - Si CHAT: répond directement sans contexte

    Args:
        query (str): La question de l'utilisateur
        llm: Instance ChatMistralAI
        rag_chain: La chaîne RAG
        max_retries (int): Nombre de tentatives en cas d'erreur 429
        initial_delay (int): Délai initial d'attente en secondes

    Returns:
        dict: Le résultat avec la réponse
    """
    from query_classifier import classify_query_intent, INTENT_RAG, INTENT_CHAT

    try:
        # Classifie l'intention
        intent = classify_query_intent(query, llm)

        if intent == INTENT_RAG:
            print(f"🔍 Mode RAG (Recherche)")
            return invoke_rag_with_retry(rag_chain, query, max_retries, initial_delay)
        else:  # INTENT_CHAT
            print(f"💬 Mode CHAT (Conversation)")
            # Réponse directe du LLM sans contexte
            result = llm.invoke(query)
            return {"answer": str(result.content)}

    except Exception as e:
        print(f"❌ Erreur lors du routing: {e}")
        return {"answer": f"Erreur: {e}"}
