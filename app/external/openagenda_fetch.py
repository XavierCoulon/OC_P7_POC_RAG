"""OpenAgenda API client for fetching events."""

import os
from datetime import datetime
from typing import Optional
from urllib.parse import quote

import requests
from pydantic import BaseModel


class LocationCoordinates(BaseModel):
    lon: float
    lat: float


class Event(BaseModel):
    uid: str
    slug: str
    canonicalurl: str
    title_fr: Optional[str] = None
    description_fr: Optional[str] = None
    longdescription_fr: Optional[str] = None
    conditions_fr: Optional[str] = None
    keywords_fr: Optional[list[str]] = None
    image: Optional[str] = None
    imagecredits: Optional[str] = None
    thumbnail: Optional[str] = None
    originalimage: Optional[str] = None
    updatedat: Optional[datetime] = None
    daterange_fr: Optional[str] = None
    firstdate_begin: Optional[datetime] = None
    firstdate_end: Optional[datetime] = None
    lastdate_begin: Optional[datetime] = None
    lastdate_end: Optional[datetime] = None
    timings: Optional[str] = None
    accessibility: Optional[str | list] = None
    accessibility_label_fr: Optional[str | list] = None
    location_uid: Optional[str] = None
    location_coordinates: Optional[LocationCoordinates] = None
    location_name: Optional[str] = None
    location_address: Optional[str] = None
    location_district: Optional[str] = None
    location_insee: Optional[str] = None
    location_postalcode: Optional[str] = None
    location_city: Optional[str] = None
    location_department: Optional[str] = None
    location_region: Optional[str] = None
    location_countrycode: Optional[str] = None
    location_image: Optional[str] = None
    location_imagecredits: Optional[str] = None
    location_phone: Optional[str] = None
    location_website: Optional[str] = None
    location_links: Optional[str] = None
    location_tags: Optional[str] = None
    location_description_fr: Optional[str] = None
    location_access_fr: Optional[str] = None
    attendancemode: Optional[str] = None
    onlineaccesslink: Optional[str] = None
    status: Optional[str] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    originagenda_title: Optional[str] = None
    originagenda_uid: Optional[str] = None
    contributor_email: Optional[str] = None
    contributor_contactnumber: Optional[str] = None
    contributor_contactname: Optional[str] = None
    contributor_contactposition: Optional[str] = None
    contributor_organization: Optional[str] = None
    category: Optional[str] = None
    country_fr: Optional[str] = None
    registration: Optional[str] = None
    links: Optional[str] = None

    class Config:
        extra = "allow"  # Allow extra fields that might be added by the API


LOCATION_DEPARTMENT = os.getenv("LOCATION_DEPARTMENT", "Pyrénées-Atlantiques")
ENCODED_DEPARTMENT = quote(LOCATION_DEPARTMENT)
FIRST_DATE = os.getenv("FIRST_DATE", "2025-01-01T00:00:00")
LIMIT = 100

BASE_URL = (
    f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    f"evenements-publics-openagenda/records"
    f"?where=location_department='{ENCODED_DEPARTMENT}' AND firstdate_begin>'{FIRST_DATE}'"
    f"&limit={LIMIT}&order_by=firstdate_begin ASC"
)


def fetch_all_events(base_url, limit=LIMIT) -> list[Event]:
    """Fetch all events by paginating through results using offset."""
    all_events: list[Event] = []
    offset = 0
    total_count = None

    while True:
        url = f"{base_url}&offset={offset}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to retrieve data: {response.status_code}")
            break

        data = response.json()

        # Get total count on first request
        if total_count is None:
            total_count = data.get("total_count", 0)

        # Get results from current page
        results = data.get("results", [])
        if not results:
            break

        # Parse and validate events
        events = [Event(**event) for event in results]
        all_events.extend(events)

        # Stop if we've fetched all events
        if len(all_events) >= total_count:
            break

        offset += limit

    return all_events


if __name__ == "__main__":
    events = fetch_all_events(BASE_URL)
    print(f"\nTotal events fetched: {len(events)}")

    # Verify no duplicates
    unique_uids = set()
    duplicates = 0
    for event in events:
        event_uid = event.uid
        if event_uid in unique_uids:
            duplicates += 1
        else:
            unique_uids.add(event_uid)

    print(f"Unique events: {len(unique_uids)}")
    print(f"Duplicate events: {duplicates}")
