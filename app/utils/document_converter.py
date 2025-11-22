"""Document processing and conversion utilities."""

import re
import time
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.external.openagenda_fetch import Event


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
    """Chunk an event document into smaller pieces."""
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
    """Embed texts with retry logic for rate limiting."""
    for attempt in range(retries):
        try:
            return embedding_model.embed_documents(texts)
        except Exception as e:
            if "429" in str(e):
                time.sleep(delay * (attempt + 1))
                continue
            raise e
    raise RuntimeError("Max retry reached for embeddings")
