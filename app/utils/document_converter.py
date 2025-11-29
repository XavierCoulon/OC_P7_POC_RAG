"""Document processing and conversion utilities."""

import copy
import re
from typing import Any, Dict, List

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.external.openagenda_fetch import Event


def format_keywords(keywords_input) -> str:
    """Format keywords as a clean string."""
    if not keywords_input:
        return ""
    if isinstance(keywords_input, list):
        return ", ".join(str(kw).strip() for kw in keywords_input if kw)
    if ";" in keywords_input:
        kws = keywords_input.split(";")
        return ", ".join(kw.strip().rstrip(".") for kw in kws if kw.strip())
    return str(keywords_input).strip()


def clean_html_content(html_content: str) -> str:
    """Cleans HTML content and returns plain text."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def normalize_whitespace(text: str) -> str:
    """Normalizes whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def get_iso_date(date_obj: Any) -> str:
    """
    Converts any date to 'YYYY-MM-DD' string format.
    Handles None values and removes time/timezone to avoid LLM confusion.
    """
    if not date_obj:
        return ""
    # On convertit en string et on garde les 10 premiers chars (2025-06-21)
    return str(date_obj)[:10]


class DocumentBuilder:
    """Builder for converting Events to chunked LangChain Documents."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # On garde les séparateurs par défaut mais on s'assure que le split est propre
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            keep_separator=False,
        )

    def _build_content(self, event: Event) -> str:
        """Build structured page_content from Event data."""
        title = event.title_fr or "Titre inconnu"
        short_desc = event.description_fr or ""
        long_desc = clean_html_content(event.longdescription_fr or "")
        keywords = format_keywords(event.keywords_fr)

        # Construction géographique complète
        loc_parts = [
            event.location_name,
            event.location_address,
            f"{event.location_city} ({event.location_postalcode})",
            event.location_department,
        ]
        location_full = ", ".join([p for p in loc_parts if p])
        date_readable = event.daterange_fr or "Dates non spécifiées"
        date_iso = f"Début: {get_iso_date(event.firstdate_begin)} / Fin: {get_iso_date(event.lastdate_end)}"
        page_content = f"""Titre: {title}
Lieu: {location_full}
Dates (Texte): {date_readable}
Dates (ISO): {date_iso}

Description:
{short_desc}

Détails:
{long_desc}

Mots-clés: {keywords}
URL: {event.canonicalurl}
"""
        return normalize_whitespace(page_content)

    def _build_metadata(self, event: Event) -> Dict[str, Any]:
        """Build metadata from Event data."""
        return {
            # --- Identifiants ---
            # L'ID unique de l'événement (ex: "37280107")
            "uid": event.uid,
            # L'ID de l'agenda source (ex: "38495884") - Correction ici (uid au lieu de id)
            "originagenda_uid": event.originagenda_uid,
            # --- Informations Essentielles (POUR LE RAG) ---
            "title": event.title_fr,
            # Le nom de l'agenda source (ex: "Mes événements France Travail")
            "originagenda_title": event.originagenda_title,
            "canonicalurl": event.canonicalurl,
            # --- Filtres Géographiques ---
            "location_name": event.location_name,
            "location_city": event.location_city,
            "location_postalcode": event.location_postalcode,
            "location_department": event.location_department,
            # --- Filtres Temporels ---
            "firstdate_begin": get_iso_date(event.firstdate_begin),
            "lastdate_end": get_iso_date(event.lastdate_end),
            # --- Informations Complémentaires ---
            "conditions_fr": event.conditions_fr,
            # --- Mots clés ---
            "keywords": format_keywords(event.keywords_fr),
        }

    def build(self, event: Event) -> List[Document]:
        """Convert Event to chunked LangChain Documents."""

        # 1. Création du texte complet
        page_content = self._build_content(event)

        # 2. Création des métadonnées de base
        base_metadata = self._build_metadata(event)

        # 3. Découpage (Splitting)
        chunks_text = self.splitter.split_text(page_content)

        documents = []
        for i, chunk_text in enumerate(chunks_text):
            # 4. Création d'une copie des métadonnées pour ce chunk
            # On ajoute chunk_index pour pouvoir re-trier si besoin
            chunk_metadata = copy.deepcopy(base_metadata)
            chunk_metadata["chunk_index"] = i

            # Création du document
            doc = Document(page_content=chunk_text, metadata=chunk_metadata)
            documents.append(doc)

        return documents


def event_to_langchain_document(event: Event) -> List[Document]:
    """Convert an Event to LangChain Documents."""
    builder = DocumentBuilder()
    return builder.build(event)
