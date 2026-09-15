import httpx
import pytest

from rag_poc.fetch import fetch_page


def test_fetch_page_builds_content_api_url(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"title": "Private renting"})

    monkeypatch.setattr(
        "rag_poc.fetch._client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )

    result = fetch_page("private-renting")

    assert seen["url"] == "https://www.gov.uk/api/content/private-renting"
    assert result == {"title": "Private renting"}


def test_fetch_page_raises_on_non_200(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    monkeypatch.setattr(
        "rag_poc.fetch._client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(httpx.HTTPStatusError):
        fetch_page("does-not-exist")
