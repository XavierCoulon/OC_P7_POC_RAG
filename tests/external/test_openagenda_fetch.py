"""Tests for OpenAgenda API client."""

from datetime import datetime
from unittest.mock import Mock, patch

from app.external.openagenda_fetch import (
    BASE_URL,
    ENCODED_DEPARTMENT,
    FIRST_DATE,
    LIMIT,
    LOCATION_DEPARTMENT,
    Event,
    fetch_all_events,
)


class TestEventModel:
    """Test Event Pydantic model."""

    def test_event_with_optional_fields(self):
        """Test creating event with optional fields."""
        event = Event(
            uid="123",
            slug="test",
            canonicalurl="https://example.com",
            title_fr="Concert",
            description_fr="Un concert sympa",
            location_city="Paris",
            location_department="Île-de-France",
        )
        assert event.title_fr == "Concert"
        assert event.location_city == "Paris"
        assert event.location_department == "Île-de-France"
        # Other optional fields should be None
        assert event.age_min is None
        assert event.accessibility is None

    def test_event_with_dates(self):
        """Test event validates datetime fields correctly."""
        date = datetime(2025, 6, 21, 20, 0, 0)
        event = Event(
            uid="123",
            slug="test",
            canonicalurl="https://example.com",
            firstdate_begin=date,
            lastdate_end=date,
        )
        assert event.firstdate_begin == date
        assert event.lastdate_end == date

    def test_event_with_keywords_list(self):
        """Test event with keywords list."""
        keywords = ["musique", "concert", "jazz"]
        event = Event(
            uid="123",
            slug="test",
            canonicalurl="https://example.com",
            keywords_fr=keywords,
        )
        assert event.keywords_fr == keywords


class TestConfigurationConstants:
    """Test module configuration constants."""

    def test_configuration_constants_loaded(self):
        """Test that all configuration constants are properly loaded."""
        assert LOCATION_DEPARTMENT == "Pyrénées-Atlantiques"
        assert ENCODED_DEPARTMENT == "Pyr%C3%A9n%C3%A9es-Atlantiques"
        assert FIRST_DATE == "2025-01-01T00:00:00"
        assert LIMIT == 100

    def test_base_url_contains_all_parameters(self):
        """Test that BASE_URL includes all required query parameters."""
        assert "evenements-publics-openagenda" in BASE_URL
        assert ENCODED_DEPARTMENT in BASE_URL
        assert FIRST_DATE in BASE_URL
        assert f"limit={LIMIT}" in BASE_URL
        assert "order_by=firstdate_begin ASC" in BASE_URL


class TestFetchAllEvents:
    """Test fetch_all_events function."""

    @patch("app.external.openagenda_fetch.requests.get")
    def test_fetch_all_events_single_page(self, mock_get):
        """Test fetching events from single page response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_count": 2,
            "results": [
                {
                    "uid": "event1",
                    "slug": "event-1",
                    "canonicalurl": "https://example.com/event1",
                    "title_fr": "Concert",
                    "location_city": "Paris",
                },
                {
                    "uid": "event2",
                    "slug": "event-2",
                    "canonicalurl": "https://example.com/event2",
                    "title_fr": "Festival",
                    "location_city": "Lyon",
                },
            ],
        }
        mock_get.return_value = mock_response

        events = fetch_all_events(BASE_URL, limit=LIMIT)

        assert len(events) == 2
        assert events[0].uid == "event1"
        assert events[0].title_fr == "Concert"
        assert events[1].uid == "event2"

    @patch("app.external.openagenda_fetch.requests.get")
    def test_fetch_all_events_multiple_pages_pagination(self, mock_get):
        """Test fetching events across multiple pages with offset pagination."""
        # Create 2 page responses - fetch stops when total_count is reached
        page1_response = Mock()
        page1_response.status_code = 200
        page1_response.json.return_value = {
            "total_count": 150,  # Total events available
            "results": [
                {
                    "uid": f"event{i}",
                    "slug": f"event-{i}",
                    "canonicalurl": f"https://example.com/event{i}",
                }
                for i in range(1, 101)  # First 100 events
            ],
        }

        page2_response = Mock()
        page2_response.status_code = 200
        page2_response.json.return_value = {
            "total_count": 150,
            "results": [
                {
                    "uid": f"event{i}",
                    "slug": f"event-{i}",
                    "canonicalurl": f"https://example.com/event{i}",
                }
                for i in range(101, 151)  # Last 50 events
            ],
        }

        mock_get.side_effect = [page1_response, page2_response]

        events = fetch_all_events(BASE_URL, limit=100)

        # Verify all events were fetched
        assert len(events) == 150
        assert mock_get.call_count == 2
        # Verify offset parameters in calls
        assert "offset=0" in mock_get.call_args_list[0][0][0]
        assert "offset=100" in mock_get.call_args_list[1][0][0]

    @patch("app.external.openagenda_fetch.requests.get")
    def test_fetch_all_events_empty_results(self, mock_get):
        """Test fetching when no events are returned."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"total_count": 0, "results": []}
        mock_get.return_value = mock_response

        events = fetch_all_events(BASE_URL)

        assert len(events) == 0

    @patch("app.external.openagenda_fetch.requests.get")
    def test_fetch_all_events_stops_on_empty_page(self, mock_get):
        """Test that fetching stops when empty results are received."""
        # First page with events
        page1_response = Mock()
        page1_response.status_code = 200
        page1_response.json.return_value = {
            "total_count": 100,
            "results": [
                {
                    "uid": f"event{i}",
                    "slug": f"event-{i}",
                    "canonicalurl": f"https://example.com/event{i}",
                }
                for i in range(1, 101)
            ],
        }

        # Empty second page (function stops on empty results)
        page2_response = Mock()
        page2_response.status_code = 200
        page2_response.json.return_value = {"total_count": 100, "results": []}

        mock_get.side_effect = [page1_response, page2_response]

        events = fetch_all_events(BASE_URL, limit=100)

        # When 100 events fetched and total_count is 100, function stops without second call
        assert len(events) == 100
        # Since len(all_events) == total_count after first page, no second call is made
        assert mock_get.call_count == 1

    @patch("app.external.openagenda_fetch.requests.get")
    def test_fetch_all_events_api_error_handling(self, mock_get):
        """Test handling of API errors (non-200 status)."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        events = fetch_all_events(BASE_URL)

        assert len(events) == 0

    @patch("app.external.openagenda_fetch.requests.get")
    def test_fetch_all_events_validates_event_objects(self, mock_get):
        """Test that returned events are valid Event Pydantic objects."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_count": 1,
            "results": [
                {
                    "uid": "event1",
                    "slug": "event-1",
                    "canonicalurl": "https://example.com",
                    "title_fr": "Concert",
                    "location_city": "Pau",
                    "location_department": "Pyrénées-Atlantiques",
                    "firstdate_begin": "2025-06-21T20:00:00",
                }
            ],
        }
        mock_get.return_value = mock_response

        events = fetch_all_events(BASE_URL)

        assert len(events) == 1
        assert isinstance(events[0], Event)
        assert events[0].uid == "event1"
        assert events[0].title_fr == "Concert"
        assert events[0].location_city == "Pau"

    @patch("app.external.openagenda_fetch.requests.get")
    def test_fetch_all_events_handles_missing_optional_fields(self, mock_get):
        """Test that events with missing optional fields default to None."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "total_count": 1,
            "results": [
                {
                    "uid": "event1",
                    "slug": "event-1",
                    "canonicalurl": "https://example.com",
                    # Optional fields are missing
                }
            ],
        }
        mock_get.return_value = mock_response

        events = fetch_all_events(BASE_URL)

        assert len(events) == 1
        assert events[0].uid == "event1"
        assert events[0].title_fr is None
        assert events[0].description_fr is None
        assert events[0].location_city is None
