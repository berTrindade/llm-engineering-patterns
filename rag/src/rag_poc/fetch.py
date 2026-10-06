"""GOV.UK Content API client."""

import httpx

GOVUK_API_BASE = "https://www.gov.uk/api/content"

PAGES = [
    "private-renting",
    "tenancy-deposit-protection",
    "deposit-protection-schemes-and-landlords",
    "renting-out-a-property",
    "evicting-tenants",
    "check-tenant-right-to-rent-documents",
]


def _client() -> httpx.Client:
    return httpx.Client()


def fetch_page(path: str, timeout: float = 30.0) -> dict:
    with _client() as client:
        response = client.get(f"{GOVUK_API_BASE}/{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()
