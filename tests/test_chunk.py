from rag_poc.chunk import chunk_page, chunk_part, html_to_text


def test_html_to_text_strips_tags_and_decodes_entities():
    html = "<p>Landlords &amp; tenants <b>must</b> agree.</p>"
    assert html_to_text(html) == "Landlords & tenants must agree."


def test_html_to_text_collapses_whitespace():
    html = "<p>Line one</p>\n\n<p>Line   two</p>"
    text = html_to_text(html)
    assert "Line one" in text
    assert "Line   two" not in text  # internal runs collapsed
    assert "Line two" in text


def test_chunk_part_splits_on_headings():
    html = (
        "<h2 id='what'>What your landlord must do</h2>"
        "<p>Fix the boiler.</p>"
        "<h2 id='you'>Your responsibilities</h2>"
        "<p>Keep it clean.</p>"
    )
    chunks = chunk_part(
        part_html=html,
        page_title="Private renting",
        part_title="Repairs",
        source_url="https://www.gov.uk/private-renting/repairs",
    )

    assert len(chunks) == 2
    assert chunks[0].heading_path == "Private renting > Repairs > What your landlord must do"
    assert "Fix the boiler" in chunks[0].text
    assert chunks[1].heading_path == "Private renting > Repairs > Your responsibilities"
    assert "Keep it clean" in chunks[1].text
    assert chunks[0].source_url == "https://www.gov.uk/private-renting/repairs"
    assert chunks[0].id != chunks[1].id


def test_chunk_part_uses_part_title_when_no_heading_present():
    html = "<p>Only one paragraph, no heading at all.</p>"
    chunks = chunk_part(
        part_html=html,
        page_title="Private renting",
        part_title="Overview",
        source_url="https://www.gov.uk/private-renting/overview",
    )
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Private renting > Overview"


def test_chunk_part_splits_oversized_section_with_overlap():
    long_para = " ".join(f"sentence{i}." for i in range(200))
    html = f"<h2 id='big'>Big section</h2><p>{long_para}</p>"

    chunks = chunk_part(
        part_html=html,
        page_title="Page",
        part_title="Part",
        source_url="https://example.test/page",
        max_chars=200,
        overlap_chars=50,
    )

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 200 + 50  # cap plus overlap slack
    # overlap: end of chunk 0 should share text with the start of chunk 1
    tail = chunks[0].text[-30:]
    assert any(word in chunks[1].text for word in tail.split()[:3])


def test_chunk_page_builds_source_url_from_base_path_and_part_slug():
    page_json = {
        "title": "Private renting",
        "base_path": "/private-renting",
        "details": {
            "parts": [
                {
                    "slug": "repairs",
                    "title": "Repairs",
                    "body": "<h2 id='x'>What to do</h2><p>Contact your landlord.</p>",
                },
                {
                    "slug": "deposits",
                    "title": "Deposits",
                    "body": "<h2 id='y'>Protection</h2><p>Deposits must be protected.</p>",
                },
            ]
        },
    }

    chunks = chunk_page(page_json)

    assert len(chunks) == 2
    urls = {c.source_url for c in chunks}
    assert urls == {
        "https://www.gov.uk/private-renting/repairs",
        "https://www.gov.uk/private-renting/deposits",
    }
    assert all(c.page_title == "Private renting" for c in chunks)


def test_chunk_page_handles_page_without_parts():
    page_json = {
        "title": "Some guidance",
        "base_path": "/some-guidance",
        "details": {"body": "<h2 id='a'>Intro</h2><p>Some text here.</p>"},
    }

    chunks = chunk_page(page_json)

    assert len(chunks) == 1
    assert chunks[0].source_url == "https://www.gov.uk/some-guidance"
    assert chunks[0].heading_path == "Some guidance > Intro"
