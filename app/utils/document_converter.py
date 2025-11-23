"""Document processing and conversion utilities."""

import re
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.external.openagenda_fetch import Event


def format_keywords(keywords_input) -> str:
    """Format keywords as a clean string "keyword1, keyword2, ...".

    Handles various input formats:
    - List of strings: ['bricolage', 'jardinage'] -> "bricolage, jardinage"
    - Semicolon-separated: 'bricothèque;bricolage;diy.' -> "bricothèque, bricolage, diy"
    - Already a string: 'bricolage, jardinage' -> "bricolage, jardinage"

    Args:
        keywords_input: Keywords in any supported format

    Returns:
        str: Clean comma-separated keywords string
    """
    if not keywords_input:
        return ""

    # If it's a list, join directly
    if isinstance(keywords_input, list):
        return ", ".join(str(kw).strip() for kw in keywords_input if kw)

    # If it's a string with semicolons, split and rejoin with commas
    if ";" in keywords_input:
        kws = keywords_input.split(";")
        return ", ".join(kw.strip().rstrip(".") for kw in kws if kw.strip())

    # Otherwise return as-is
    return str(keywords_input).strip()


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


def _safe_str_date(date_obj) -> str | None:
    """Safely convert datetime to string, handling None values.

    Args:
        date_obj: datetime object or None

    Returns:
        String representation or None
    """
    return str(date_obj) if date_obj else None


class DocumentBuilder:
    """Builder for converting Events to chunked LangChain Documents.

    Encapsulates the logic for:
    - Building rich page_content from Event data
    - Chunking content for optimal embedding
    - Creating metadata
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """Initialize DocumentBuilder.

        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def _build_content(self, event: Event) -> str:
        """Build structured page_content from Event data.

        Args:
            event: The event to build content from

        Returns:
            Formatted page_content string
        """
        title = event.title_fr or ""
        short_desc = event.description_fr or ""
        long_desc = clean_html_content(event.longdescription_fr or "")
        keywords = format_keywords(event.keywords_fr)
        location_name = event.location_name or ""
        city = event.location_city or ""
        dept = event.location_department or ""
        region = event.location_region or ""
        address = event.location_address or ""
        date_range = event.daterange_fr or ""
        start = event.firstdate_begin or ""
        end = event.firstdate_end or ""

        page_content = f"""Titre: {title}

Description: {short_desc}

Détails: {long_desc}

Localisation: {location_name}, {city} ({dept}), {region}
Adresse: {address}

Mots-clés: {keywords}

Dates: {date_range}
Début: {start}
Fin: {end}"""

        return normalize_whitespace(page_content)

    def _build_metadata(self, event: Event) -> dict:
        """Build metadata from Event data.

        Args:
            event: The event to extract metadata from

        Returns:
            Metadata dictionary
        """
        return {
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
            "firstdate_begin": _safe_str_date(event.firstdate_begin),
            "firstdate_end": _safe_str_date(event.firstdate_end),
            "lastdate_begin": _safe_str_date(event.lastdate_begin),
            "lastdate_end": _safe_str_date(event.lastdate_end),
            # Source info
            "originagenda_title": event.originagenda_title,
            "originagenda_uid": event.originagenda_uid,
            # Event metadata
            "age_min": event.age_min,
            "age_max": event.age_max,
        }

    def build(self, event: Event) -> list[Document]:
        """Convert Event to chunked LangChain Documents.

        Args:
            event: The event to convert

        Returns:
            List of Document objects (one per chunk)
        """
        page_content = self._build_content(event)
        chunks = self.splitter.split_text(page_content)
        metadata = self._build_metadata(event)

        return [Document(page_content=chunk, metadata=metadata) for chunk in chunks]


def event_to_langchain_document(event: Event) -> list[Document]:
    """Convert an Event to LangChain Documents optimized for RAG with chunking.

    Convenience function that creates a DocumentBuilder and builds documents.

    Args:
        event (Event): The event to convert.

    Returns:
        list[Document]: List of LangChain Documents (chunked).
    """
    builder = DocumentBuilder()
    return builder.build(event)
