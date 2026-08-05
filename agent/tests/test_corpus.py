"""The Corpus manifest, the licence data that travels with every chunk, and the
ingest command. No network: the fetcher is an argument."""

from __future__ import annotations

import pytest

from nutrigraph_agent.corpus import (
    CHUNK_CHARS,
    FORBIDDEN_LICENCES,
    LICENCES,
    CorpusEntry,
    LicenceRefused,
    check_licences,
    chunks,
    content_hash,
    html_sections,
    load_manifest,
)

MANIFEST = load_manifest()


# --- what may be in the Corpus ------------------------------------------------


def test_the_corpus_is_about_forty_documents():
    assert 35 <= len(MANIFEST) <= 50


def test_no_efsa_prose_is_in_the_corpus():
    """CC BY-ND does not license the adapted material that chunking prose into
    an index arguably produces, so a European reference value enters as a number
    in a data table instead, and never as a chunk."""
    for entry in MANIFEST:
        assert "efsa" not in entry.source_url.lower(), entry.slug
        assert entry.licence_id not in FORBIDDEN_LICENCES


def test_who_stays_in_and_is_marked_non_commercial():
    who = [e for e in MANIFEST if e.publisher == "World Health Organization"]

    assert who, "WHO stays in; this demonstration is not commercial"
    for entry in who:
        assert entry.licence.commercial_use is False
        assert "CC BY-NC-SA 3.0 IGO" in entry.attribution
        # WHO's own required wording, with the document and the year in it.
        assert entry.title in entry.attribution


def test_every_document_carries_a_known_licence_and_an_attribution_string():
    for entry in MANIFEST:
        assert entry.licence_id in LICENCES, entry.slug
        assert entry.attribution.strip(), entry.slug


def test_a_no_derivatives_licence_is_refused_at_load():
    with pytest.raises(LicenceRefused, match="no-derivatives"):
        check_licences(
            [
                CorpusEntry(
                    slug="efsa-drv-iron",
                    title="Dietary Reference Values for iron",
                    source_url="https://www.efsa.europa.eu/en/efsajournal/pub/4254",
                    publisher="EFSA",
                    licence_id="cc-by-nd",
                )
            ]
        )


def test_the_slugs_are_unique():
    slugs = [entry.slug for entry in MANIFEST]

    assert len(set(slugs)) == len(slugs)


# --- turning a document into passages -----------------------------------------

PAGE = """
<html><head><style>p { color: red }</style></head><body>
<nav><a href="/">skip me</a></nav>
<h1>Protein</h1>
<p>Protein foods supply the amino acids the body cannot make for itself, and
they include eggs, seafood, lean meats, beans, peas, lentils, and nuts.</p>
<h2>How much</h2>
<ul><li>Adults need a variety of protein foods spread across the day rather than
in one large serving at dinner, which is the usual pattern.</li></ul>
<script>console.log("skip me too")</script>
</body></html>
"""


def test_a_page_becomes_sections_named_by_their_heading():
    sections = html_sections(PAGE)

    assert [heading for heading, _ in sections] == ["Protein", "How much"]
    assert "amino acids" in sections[0][1]


def test_navigation_and_scripts_are_not_passages():
    body = " ".join(text for _, text in html_sections(PAGE))

    assert "skip me" not in body
    assert "console.log" not in body


def test_a_chunk_keeps_the_locator_and_stays_within_the_size():
    long_section = [("page 4", "A sentence about sodium. " * 400)]

    produced = chunks(long_section)

    assert len(produced) > 1
    assert {locator for locator, _ in produced} == {"page 4"}
    assert all(len(text) <= CHUNK_CHARS for _, text in produced)


def test_a_navigation_crumb_is_too_short_to_be_a_chunk():
    assert chunks([("Home", "Skip to main content")]) == []


def test_the_content_hash_is_what_makes_a_second_run_cheap():
    passages = [("page 1", "a" * 200), ("page 2", "b" * 200)]

    assert content_hash(passages) == content_hash(list(passages))
    assert content_hash(passages) != content_hash(passages[:1])
