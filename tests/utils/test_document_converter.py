"""Tests for document converter utilities."""

from datetime import datetime

import pytest
from langchain_core.documents import Document

from app.external.openagenda_fetch import Event
from app.utils.document_converter import (
    DocumentBuilder,
    clean_html_content,
    event_to_langchain_document,
    format_keywords,
    get_iso_date,
    normalize_whitespace,
)


class TestFormatKeywords:
    """Tests for format_keywords function."""

    def test_format_keywords_from_list(self):
        """Test formatting keywords from a list."""
        keywords = ["bricolage", "jardinage", "diy"]
        result = format_keywords(keywords)
        assert result == "bricolage, jardinage, diy"

    def test_format_keywords_from_semicolon_string(self):
        """Test formatting keywords from semicolon-separated string."""
        keywords = "bricothèque;bricolage;diy."
        result = format_keywords(keywords)
        assert result == "bricothèque, bricolage, diy"

    def test_format_keywords_from_comma_string(self):
        """Test formatting keywords already comma-separated."""
        keywords = "bricolage, jardinage, diy"
        result = format_keywords(keywords)
        assert result == "bricolage, jardinage, diy"

    def test_format_keywords_empty_list(self):
        """Test formatting empty keywords list."""
        result = format_keywords([])
        assert result == ""

    def test_format_keywords_none(self):
        """Test formatting None keywords."""
        result = format_keywords(None)
        assert result == ""

    def test_format_keywords_with_empty_strings(self):
        """Test formatting keywords with empty strings."""
        keywords = ["bricolage", "", "diy"]
        result = format_keywords(keywords)
        assert result == "bricolage, diy"

    def test_format_keywords_single_keyword(self):
        """Test formatting single keyword."""
        result = format_keywords(["bricolage"])
        assert result == "bricolage"


class TestCleanHtmlContent:
    """Tests for clean_html_content function."""

    def test_clean_html_basic(self):
        """Test cleaning basic HTML."""
        html = "<p>Hello <strong>World</strong></p>"
        result = clean_html_content(html)
        assert result == "Hello World"

    def test_clean_html_with_tags(self):
        """Test cleaning HTML with various tags."""
        html = "<div><h1>Title</h1><p>Content</p></div>"
        result = clean_html_content(html)
        assert "Title" in result and "Content" in result

    def test_clean_html_preserves_text_separation(self):
        """Test that cleaning preserves text separation."""
        html = "<p>First</p><p>Second</p>"
        result = clean_html_content(html)
        assert "First" in result and "Second" in result

    def test_clean_html_empty_string(self):
        """Test cleaning empty HTML."""
        result = clean_html_content("")
        assert result == ""

    def test_clean_html_with_entities(self):
        """Test cleaning HTML with entities."""
        html = "<p>&nbsp;Test&nbsp;</p>"
        result = clean_html_content(html)
        assert "Test" in result


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace function."""

    def test_normalize_multiple_spaces(self):
        """Test normalizing multiple spaces."""
        text = "Hello    World"
        result = normalize_whitespace(text)
        assert result == "Hello World"

    def test_normalize_tabs_and_newlines(self):
        """Test normalizing tabs and newlines."""
        text = "Hello\n\n\tWorld"
        result = normalize_whitespace(text)
        assert result == "Hello World"

    def test_normalize_leading_trailing_spaces(self):
        """Test removing leading and trailing spaces."""
        text = "   Hello World   "
        result = normalize_whitespace(text)
        assert result == "Hello World"

    def test_normalize_mixed_whitespace(self):
        """Test normalizing mixed whitespace."""
        text = "  Hello \n  World  \t Test  "
        result = normalize_whitespace(text)
        assert result == "Hello World Test"

    def test_normalize_no_extra_whitespace(self):
        """Test text with no extra whitespace."""
        text = "Hello World"
        result = normalize_whitespace(text)
        assert result == "Hello World"


class TestSafeStrDate:
    """Tests for get_iso_date function."""

    def testget_iso_date_with_datetime(self):
        """Test converting datetime to string."""
        date = datetime(2025, 1, 15)
        result = get_iso_date(date)
        assert result is not None
        assert "2025" in result and "01" in result and "15" in result

    def testget_iso_date_with_none(self):
        """Test converting None to empty string."""
        result = get_iso_date(None)
        assert result == ""

    def testget_iso_date_with_string(self):
        """Test converting string (should work as it's converted to string)."""
        result = get_iso_date("2025-01-15")
        assert result == "2025-01-15"


class TestDocumentBuilder:
    """Tests for DocumentBuilder class."""

    @pytest.fixture
    def sample_event(self):
        """Create a sample Event for testing."""
        return Event(
            uid="event-123",
            slug="test-event",
            title_fr="Atelier de Bricolage",
            description_fr="Un atelier amusant",
            longdescription_fr="<p>Apprenez à bricoler avec nos experts</p>",
            keywords_fr=["bricolage", "diy"],
            location_name="Centre Culturel",
            location_city="Pau",
            location_department="Pyrénées-Atlantiques",
            location_region="Aquitaine",
            location_address="123 Rue de la Paix",
            location_postalcode="64000",
            canonicalurl="https://example.com/event",
            daterange_fr="15-17 janvier 2025",
            firstdate_begin=datetime(2025, 1, 15),
            firstdate_end=datetime(2025, 1, 17),
            lastdate_begin=None,
            lastdate_end=None,
            conditions_fr="Gratuit",
            originagenda_title="Agenda Test",
            originagenda_uid="agenda-123",
            age_min=0,
            age_max=None,
        )

    def test_document_builder_initialization(self):
        """Test DocumentBuilder initialization."""
        builder = DocumentBuilder(chunk_size=600, chunk_overlap=100)
        assert builder.chunk_size == 600
        assert builder.chunk_overlap == 100

    def test_document_builder_build_content(self, sample_event):
        """Test building content from event."""
        builder = DocumentBuilder()
        content = builder._build_content(sample_event)

        assert "Atelier de Bricolage" in content
        assert "Un atelier amusant" in content
        assert "Pau" in content
        assert "bricolage, diy" in content

    def test_document_builder_build_metadata(self, sample_event):
        """Test building metadata from event."""
        builder = DocumentBuilder()
        metadata = builder._build_metadata(sample_event)

        assert metadata["uid"] == "event-123"
        assert metadata["location_city"] == "Pau"
        assert metadata["location_department"] == "Pyrénées-Atlantiques"
        assert "2025" in metadata["firstdate_begin"]

    def test_document_builder_build_creates_documents(self, sample_event):
        """Test that build creates valid Documents."""
        builder = DocumentBuilder()
        documents = builder.build(sample_event)

        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)

        for doc in documents:
            assert len(doc.page_content) > 0
            assert doc.metadata is not None
            assert "uid" in doc.metadata

    def test_document_builder_chunking(self):
        """Test that DocumentBuilder chunks content correctly."""
        builder = DocumentBuilder(chunk_size=100, chunk_overlap=10)
        event = Event(
            uid="long-event",
            slug="long-event",
            title_fr="Long Event",
            description_fr="x" * 500,  # 500 characters
            longdescription_fr="",
            keywords_fr=[],
            location_name="Test",
            location_city="Pau",
            location_department="Pyrénées-Atlantiques",
            location_region="Aquitaine",
            location_address="Address",
            location_postalcode="64000",
            canonicalurl="https://example.com",
            daterange_fr="",
            firstdate_begin=None,
            firstdate_end=None,
            lastdate_begin=None,
            lastdate_end=None,
            originagenda_title="Agenda",
            originagenda_uid="agenda",
            age_min=None,
            age_max=None,
        )

        documents = builder.build(event)
        assert len(documents) > 1  # Should be chunked into multiple documents


class TestEventToLangchainDocument:
    """Tests for event_to_langchain_document function."""

    def test_event_to_document_basic(self):
        """Test basic event to document conversion."""
        event = Event(
            uid="event-456",
            slug="test",
            title_fr="Test Event",
            description_fr="Test description",
            longdescription_fr="",
            keywords_fr=[],
            location_name="Test Place",
            location_city="Bayonne",
            location_department="Pyrénées-Atlantiques",
            location_region="Aquitaine",
            location_address="123 Rue Test",
            location_postalcode="64100",
            canonicalurl="https://example.com",
            daterange_fr="",
            firstdate_begin=None,
            firstdate_end=None,
            lastdate_begin=None,
            lastdate_end=None,
            originagenda_title="Test Agenda",
            originagenda_uid="agenda",
            age_min=None,
            age_max=None,
        )

        documents = event_to_langchain_document(event)
        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)
