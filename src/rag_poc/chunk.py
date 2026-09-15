"""HTML govspeak body -> Chunk list, split by h2/h3 heading with a size cap and overlap."""

import re
from html.parser import HTMLParser

from pydantic import BaseModel

HEADING_TAGS = {"h2", "h3"}
GOVUK_ORIGIN = "https://www.gov.uk"


class Chunk(BaseModel):
    id: str
    text: str
    source_url: str
    page_title: str
    heading_path: str


class _Section:
    def __init__(self, heading: str | None):
        self.heading = heading
        self.parts: list[str] = []

    def text(self) -> str:
        return _collapse_whitespace(" ".join(self.parts))


class _HeadingSplitter(HTMLParser):
    """Splits an HTML fragment into sections, one per h2/h3, buffering plain text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections: list[_Section] = [_Section(heading=None)]
        self._in_heading = False
        self._heading_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in HEADING_TAGS:
            self._in_heading = True
            self._heading_text = []

    def handle_endtag(self, tag):
        if tag in HEADING_TAGS and self._in_heading:
            self._in_heading = False
            heading = _collapse_whitespace("".join(self._heading_text))
            self.sections.append(_Section(heading=heading))

    def handle_data(self, data):
        if self._in_heading:
            self._heading_text.append(data)
        else:
            self.sections[-1].parts.append(data)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def html_to_text(html: str) -> str:
    parser = _HeadingSplitter()
    parser.feed(html)
    return _collapse_whitespace(" ".join(s.text() for s in parser.sections))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _split_with_overlap(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap_chars
    return pieces


def chunk_part(
    part_html: str,
    page_title: str,
    part_title: str,
    source_url: str,
    max_chars: int = 1000,
    overlap_chars: int = 150,
) -> list[Chunk]:
    parser = _HeadingSplitter()
    parser.feed(part_html)

    chunks: list[Chunk] = []
    for section in parser.sections:
        text = section.text()
        if not text:
            continue

        path_segments = [page_title, part_title, section.heading]
        heading_path = " > ".join(seg for seg in path_segments if seg)
        base_slug = _slugify(section.heading or part_title)

        for i, piece in enumerate(_split_with_overlap(text, max_chars, overlap_chars)):
            chunk_id = f"{source_url}#{base_slug}" + (f"-{i}" if i else "")
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    source_url=source_url,
                    page_title=page_title,
                    heading_path=heading_path,
                )
            )
    return chunks


def chunk_page(page_json: dict, max_chars: int = 1000, overlap_chars: int = 150) -> list[Chunk]:
    page_title = page_json["title"]
    base_path = page_json["base_path"]
    details = page_json.get("details", {})
    parts = details.get("parts")

    chunks: list[Chunk] = []
    if parts:
        for part in parts:
            source_url = f"{GOVUK_ORIGIN}{base_path}/{part['slug']}"
            chunks.extend(
                chunk_part(
                    part_html=part["body"],
                    page_title=page_title,
                    part_title=part["title"],
                    source_url=source_url,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
            )
    else:
        source_url = f"{GOVUK_ORIGIN}{base_path}"
        chunks.extend(
            chunk_part(
                part_html=details.get("body", ""),
                page_title=page_title,
                part_title="",
                source_url=source_url,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )

    return chunks
