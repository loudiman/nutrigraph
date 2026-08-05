"""The Corpus: what may be indexed, on what terms, and how a document becomes
chunks.

**Licences are data, not documentation.** Every chunk carries its licence
identifier and its attribution string, written here at ingestion time and
repeated onto the chunk row, so a licence filter never needs a join.

**EFSA prose is excluded.** CC BY-ND does not license the adapted material that
chunking a document into an index arguably produces, so no EFSA prose enters the
Corpus; a European reference value enters as a number in a data table instead,
because a numeric fact is not a copyrightable expression. `FORBIDDEN_LICENCES`
is that decision as code, and `load_manifest` refuses a document that names one.

**WHO stays in**, because this demonstration is not commercial — and the
`commercial_use` flag it is ingested under is what makes that reversible in one
SQL predicate.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

MANIFEST = Path(__file__).resolve().parents[2] / "seeds" / "corpus.json"

# A chunk large enough to answer from and small enough to cite precisely.
CHUNK_CHARS = 1200
# Below this a "chunk" is a navigation crumb, not a passage.
MIN_CHUNK_CHARS = 160

USER_AGENT = "NutriGraph/0.1 (nutrition coach demo; contact via repository)"

HEADINGS = ("h1", "h2", "h3", "h4")
BLOCKS = HEADINGS + ("p", "li", "td", "dd", "dt", "blockquote", "figcaption")


@dataclass(frozen=True)
class Licence:
    """The terms one source is used under. `attribution` is a template, because
    WHO's required string names the document and its year."""

    licence_id: str
    commercial_use: bool
    attribution: str

    def attribute(self, *, title: str, year: int | str) -> str:
        return self.attribution.format(title=title, year=year)


LICENCES: dict[str, Licence] = {
    "us-gov-public-domain": Licence(
        licence_id="us-gov-public-domain",
        commercial_use=True,
        attribution="{title}. A work of the United States federal government, "
        "public domain under 17 U.S.C. §105.",
    ),
    "ogl-v3": Licence(
        licence_id="ogl-v3",
        commercial_use=True,
        attribution="{title}. Contains public sector information licensed under "
        "the Open Government Licence v3.0.",
    ),
    "uk-crown-copyright": Licence(
        licence_id="uk-crown-copyright",
        commercial_use=True,
        attribution="{title}. © Crown copyright {year}, used under Open "
        "Government Licence family terms; verify before commercial reuse.",
    ),
    # The required string is WHO's own wording, and it travels with every chunk.
    "cc-by-nc-sa-3.0-igo": Licence(
        licence_id="cc-by-nc-sa-3.0-igo",
        commercial_use=False,
        attribution="{title}. Geneva: World Health Organization; {year}. "
        "Licence: CC BY-NC-SA 3.0 IGO.",
    ),
    "ph-gov-ra8293-s176": Licence(
        licence_id="ph-gov-ra8293-s176",
        commercial_use=False,
        attribution="{title}. Food and Nutrition Research Institute, DOST, "
        "Philippines. A Philippine Government work under RA 8293 §176: free to "
        "read and quote, but exploitation for profit needs FNRI's prior approval.",
    ),
}

# Chunking prose into an index is arguably an adaptation, which a no-derivatives
# licence does not permit. Nothing under one of these may be ingested.
FORBIDDEN_LICENCES = frozenset({"cc-by-nd", "cc-by-nd-4.0"})


class CorpusEntry(BaseModel):
    """One document in the manifest, before it has been fetched."""

    slug: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    published_on: date | None = None
    licence_id: str = Field(min_length=1)

    @property
    def licence(self) -> Licence:
        return LICENCES[self.licence_id]

    @property
    def attribution(self) -> str:
        year = self.published_on.year if self.published_on else "n.d."
        return self.licence.attribute(title=self.title, year=year)


class LicenceRefused(ValueError):
    """A manifest entry names terms the Corpus may not be built on."""


def load_manifest(path: Path = MANIFEST) -> list[CorpusEntry]:
    import json

    entries = [CorpusEntry.model_validate(row) for row in json.loads(path.read_text("utf-8"))]
    check_licences(entries)
    return entries


def check_licences(entries: Iterable[CorpusEntry]) -> None:
    for entry in entries:
        if entry.licence_id in FORBIDDEN_LICENCES:
            raise LicenceRefused(
                f"{entry.slug} is {entry.licence_id}: a no-derivatives licence does "
                f"not permit chunking prose into an index"
            )
        if entry.licence_id not in LICENCES:
            raise LicenceRefused(f"{entry.slug} names an unknown licence {entry.licence_id!r}")


# --- turning a document into passages -----------------------------------------


class _Prose(HTMLParser):
    """Headings and paragraphs in the order they appear. A nutrition fact sheet
    is prose in a handful of tags, so the standard library is the whole parser."""

    SKIP = frozenset(
        {"script", "style", "noscript", "svg", "nav", "header", "footer",
         "aside", "form", "button", "select", "template"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self._skipping = 0
        self._tag = "p"
        self._buffer: list[str] = []

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._buffer)).strip()
        if text:
            self.blocks.append((self._tag, text))
        self._buffer = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self.SKIP:
            self._skipping += 1
        elif tag in BLOCKS:
            self._flush()
            self._tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skipping = max(0, self._skipping - 1)
        elif tag in BLOCKS:
            self._flush()
            self._tag = "p"

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def html_sections(html: str) -> list[tuple[str, str]]:
    """`(section heading, prose)` pairs. The heading is half of the Citation."""
    parser = _Prose()
    parser.feed(html)
    parser.close()
    sections: list[tuple[str, str]] = []
    heading, body = "", []
    for tag, text in parser.blocks:
        if tag in HEADINGS:
            if body:
                sections.append((heading, "\n".join(body)))
                body = []
            heading = text
        else:
            body.append(text)
    if body:
        sections.append((heading, "\n".join(body)))
    return [(h or "introduction", b) for h, b in sections if b.strip()]


def pdf_sections(data: bytes) -> list[tuple[str, str]]:
    """`(page, prose)` pairs. A page number is the locator a reader can follow."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = ((n, page.extract_text() or "") for n, page in enumerate(reader.pages, 1))
    return [
        (f"page {n}", re.sub(r"[ \t]*\n[ \t]*", "\n", text).strip())
        for n, text in pages
        if text.strip()
    ]


def sections(body: bytes, content_type: str) -> list[tuple[str, str]]:
    if "pdf" in content_type.lower():
        return pdf_sections(body)
    return html_sections(body.decode("utf-8", errors="replace"))


def chunks(
    sections: Sequence[tuple[str, str]], *, size: int = CHUNK_CHARS
) -> list[tuple[str, str]]:
    """Split each section on paragraph boundaries, so a chunk reads on its own
    and keeps the locator that a Citation names."""
    out: list[tuple[str, str]] = []
    for locator, text in sections:
        buffer = ""
        for paragraph in text.split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if buffer and len(buffer) + len(paragraph) + 1 > size:
                out.append((locator, buffer))
                buffer = ""
            buffer = f"{buffer}\n{paragraph}" if buffer else paragraph
            while len(buffer) > size:  # one paragraph longer than a whole chunk
                out.append((locator, buffer[:size]))
                buffer = buffer[size:]
        if buffer:
            out.append((locator, buffer))
    return [(locator, text) for locator, text in out if len(text) >= MIN_CHUNK_CHARS]


def fetch(url: str, *, timeout: float = 120.0) -> tuple[bytes, str]:
    import httpx

    with httpx.Client(
        follow_redirects=True, timeout=timeout, headers={"user-agent": USER_AGENT}
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "")


def fetch_chunks(entry: CorpusEntry) -> list[tuple[str, str]]:
    """The default fetcher: one document in, its passages out. Passed in as an
    argument by `ingest`, so a test can supply passages without a network."""
    body, content_type = fetch(entry.source_url)
    if not content_type and entry.source_url.lower().endswith(".pdf"):
        content_type = "application/pdf"
    return chunks(sections(body, content_type))


def content_hash(passages: Sequence[tuple[str, str]]) -> str:
    """What was ingested last time. An unchanged document is not embedded again,
    which is what makes the ingest command cheap to run twice."""
    digest = sha256()
    for locator, text in passages:
        digest.update(locator.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
