# Nutrition RAG Corpus: Source Survey & Licensing (Issue #4)

Research date: 2026-08-01. Investigated against primary sources only (issuing body's own site/licence page). Every claim below is cited.

## 1. Dietary Guidelines for Americans (DGA) — USDA/HHS

- **Current edition:** *Dietary Guidelines for Americans, 2025–2030*, released 2026-01-07, "Eat Real Food" theme, hosted at [realfood.gov](https://realfood.gov/); PDF at [cdn.realfood.gov/DGA_508.pdf](https://cdn.realfood.gov/DGA_508.pdf). This **replaces** the 2020–2025 edition — do not index the old PDF as current guidance.
- **Licence/copyright:** DGA is a work of the U.S. federal government. Under [17 U.S.C. §105](https://www.copyright.gov/title17/92chap1.html#105), U.S. Government works are not subject to copyright protection in the U.S. — i.e. public domain. dietaryguidelines.gov's own policy page states DGA text, figures, graphs and tables are public domain and usable without permission ([dietaryguidelines.gov/policy-and-links](https://www.dietaryguidelines.gov/policy-and-links)). **Caveat:** photos/illustrations embedded in the DGA PDF are often licensed stock imagery, not public domain (confirmed directly — the current PDF's embedded metadata credits a 2018 Shutterstock photo by Alexander Raths). This is a text corpus, so irrelevant beyond "don't redistribute the images."
- **Redistribution in a vector index:** Permitted, no restriction, no attribution legally required (though good practice to cite).
- **Format/size:** PDF (also `DGA.pdf`); fetched and read directly. Body content is short: cover + 9 content pages (pp.1–9, dated January 2026), organized as bullet lists under section headers (protein, dairy, vegetables/fruits, fats, grains, added sugars, alcohol, sodium, life-stage guidance for infants/children/adolescents/pregnancy/lactation/older adults/chronic disease/vegetarians). Clean, well-structured text — trivial to extract, no tables/images required for the guidance text itself.
- **Freshness/cadence:** Congressionally mandated revision every 5 years (2020, 2025 cycle confirmed by the 2020→2025→2030 numbering). Just refreshed; next edition ~2031.

## 2. WHO nutrition guidance

- **Source:** WHO "Healthy diet" fact sheet / Q&A — [who.int/publications/m/item/healthy-diet-factsheet394](https://www.who.int/publications/m/item/healthy-diet-factsheet394) and [who.int/news-room/questions-and-answers/item/healthy-diet-keys-to-eating-well](https://www.who.int/news-room/questions-and-answers/item/healthy-diet-keys-to-eating-well).
- **Licence:** WHO publications are released under **CC BY-NC-SA 3.0 IGO** ([who.int/about/policies/publishing/copyright](https://www.who.int/about/policies/publishing/copyright)). Permits copying/adaptation with attribution, **non-commercial only**, and any derivative must be shared under the same licence (share-alike). Required attribution string: `"[Title]. [Place]: World Health Organization; [Year]. Licence: CC BY-NC-SA 3.0 IGO."` WHO logo use requires separate written permission.
- **Redistribution in a vector index:** Legally fine for a non-commercial demo; if the coach product is ever commercialized, this source needs a fresh look (NC clause) or a paid WHO licence.
- **Format/size:** HTML fact sheet + Q&A page, short (single-page, a few hundred words each) — trivial extraction, no PDF wrangling needed.
- **Freshness:** Fact sheets are periodically revised (no fixed public cadence found on the page itself); check the "last updated" stamp on ingest.

## 3. EFSA Dietary Reference Values (DRVs)

- **Source:** [efsa.europa.eu/en/topics/topic/dietary-reference-values](https://www.efsa.europa.eu/en/topics/topic/dietary-reference-values) — DRVs are published as scientific opinions in the **EFSA Journal**, aggregated via the "DRV Finder" tool, not one single PDF.
- **Licence:** Two layers —
  - EFSA's general website legal notice: "Re-use is authorised, provided that EFSA is acknowledged as the source of the material" ([efsa.europa.eu/en/legalnotice](https://www.efsa.europa.eu/en/legalnotice)).
  - EFSA Journal articles specifically (where the actual DRV opinions live) are published under **CC BY-ND** — attribution required, but **no derivatives/adaptations permitted** (confirmed via Wiley's EFSA Journal open-access terms). This is a real legal wrinkle: chunking a DRV opinion's text for a vector index could plausibly be read as creating "adapted material," which CC BY-ND does not license. Recommend either (a) using only short verbatim excerpts with clear sourcing, (b) restating DRV numeric values as data facts (numbers/facts are not copyrightable) rather than storing prose blocks, or (c) contacting EFSA for clarification before shipping to production.
- **Format/size:** EFSA Journal articles are individually long scientific opinions (dozens of pages each, one per nutrient — 34+ nutrients reviewed as of 2019); DRVs are also summarized in shorter "Summary report" PDFs (e.g. the UL summary, updated May 2024). Extraction effort is nontrivial (per-nutrient PDFs, scientific-paper structure with methods/references).
- **Freshness:** Rolling re-evaluation, not fixed cycle — vitamin E and iron updated 2024; manganese, folate, vitamin D, B6 in 2023 ([efsa.europa.eu/en/topics/topic/dietary-reference-values](https://www.efsa.europa.eu/en/topics/topic/dietary-reference-values)). A demo corpus would need to track per-nutrient publication dates individually.

## 4. UK Eatwell Guide (gov.uk / OHID)

- **Source:** [gov.uk/government/publications/the-eatwell-guide](https://www.gov.uk/government/publications/the-eatwell-guide).
- **Licence:** **Open Government Licence v3.0**, Crown copyright — page states "All content is available under the Open Government Licence v3.0, except where otherwise stated." OGL permits commercial and non-commercial reuse, copying, adaptation, and redistribution with attribution — no share-alike, no non-commercial restriction. This is the most permissive source in the set.
- **Redistribution in a vector index:** Fully permitted, attribution recommended ("Contains public sector information licensed under the Open Government Licence v3.0").
- **Format/size:** PDF (colour + greyscale editions), plus JPEG and EPS graphic versions, plus an HTML explainer page. The core artifact is a single-page infographic (food-group proportions), not prose — thin on extractable text by itself; pair with the NHS explainer page (below) for actual sentences.
- **Freshness:** Last updated 2024-01-02 (reference correction), originally published 2016-03-17 — i.e. the underlying dietary advice is ~10 years old, though administratively touched in 2024.

## 5. NHS Eatwell Guide explainer (companion FAQ-style page)

- **Source:** [nhs.uk/live-well/eat-well/food-guidelines-and-food-labels/the-eatwell-guide](https://www.nhs.uk/live-well/eat-well/food-guidelines-and-food-labels/the-eatwell-guide/).
- **Licence:** Crown copyright (confirmed in page footer). NHS website content typically sits under Crown copyright/OGL-family terms rather than a bespoke licence, but the page does not itself display an OGL badge — treat as Crown-copyright-with-standard-terms and verify before commercial reuse.
- **Format/size:** HTML, prose paragraphs — good, clean extraction target and reads like FAQ content (portion sizes, food groups, practical tips).
- **Freshness:** Page states "Last reviewed: 29 November 2022, Next review due: 29 November 2025" — **that review date has now passed** (today is 2026-08-01), so this page is technically overdue for NHS's own review cycle. Flag as slightly stale; still usable but note the lapsed review date in the corpus metadata.

## 6. Philippines — FNRI Nutritional Guidelines for Filipinos (NGF)

- **Source:** [fnri.dost.gov.ph — Nutritional Guidelines for Filipinos: a prescription to good nutrition](https://www.fnri.dost.gov.ph/index.php/publications/writers-pool-corner/57-food-and-nutrition/204-nutritional-guidelines-for-filipinos-a-prescription-to-good-nutrition), PDF: [fnri.dost.gov.ph/images/images/standardtools/NGF-2012.pdf](https://www.fnri.dost.gov.ph/images/images/standardtools/NGF-2012.pdf). Latest revision: 2012 (ten messages), preceded by 2000 and 1990 editions.
- **Licence:** No explicit copyright notice found on the page itself. As a Philippine government work, it falls under **Republic Act No. 8293 (Intellectual Property Code of the Philippines), Section 176**: no copyright subsists in a work of the Philippine Government, **but** "prior approval of the government agency... shall be necessary for exploitation of such work for profit, and such agency... may impose... the payment of royalties" ([Official Gazette, RA 8293](https://www.officialgazette.gov.ph/1997/06/06/republic-act-no-8293/)). So: free to read/quote, but a *commercial* nutrition-coach product embedding this text should get FNRI's prior approval; a non-commercial demo is on much safer ground.
- **Format/size:** PDF, single consolidated document (10 core messages with rationale), not tabular DRI data (that's the separate PDRI document, a 2015 seminar-series PDF at [fnri.dost.gov.ph/images/sources/SeminarSeries/41st/PHILIPPINE-DIETARY-REFERENCE-INTAKES-2015.pdf](https://fnri.dost.gov.ph/images/sources/SeminarSeries/41st/PHILIPPINE-DIETARY-REFERENCE-INTAKES-2015.pdf)). Extraction effort: low-to-moderate, standard government PDF.
- **Freshness:** 2012 edition is the latest confirmed (14 years old); no evidence found of a newer FNRI food-based dietary guideline as of this research date.

## 7. Public nutrition FAQ sets

- **USDA MyPlate resources** — [nutrition.gov/topics/basic-nutrition/myplate-resources](https://www.nutrition.gov/topics/basic-nutrition/myplate-resources), [fna.usda.gov/tn/myplate](https://www.fna.usda.gov/tn/myplate). Same public-domain basis as DGA (17 U.S.C. §105, USDA federal work). Tip sheets are short, HTML/PDF, public domain, no attribution required.
- **Nutrition.gov Expert Q&A** — [nutrition.gov/expert-q-a](https://www.nutrition.gov/expert-q-a). Also a USDA-run federal site — same public-domain basis. Good FAQ-shaped content for RAG (short Q/A pairs), trivial extraction (HTML).
- **WHO Q&A "Healthy diet: keys to eating well"** — already covered in §2 (CC BY-NC-SA 3.0 IGO).

## Recommendation: demo-sized corpus

For a single demo, use these **5 documents** (skip the EFSA DRV opinions and the UK infographic-only PDF for v1 — see rationale below):

| # | Document | Source | Licence | Format | Est. tokens |
|---|----------|--------|---------|--------|-------------|
| 1 | Dietary Guidelines for Americans, 2025–2030 | [cdn.realfood.gov/DGA_508.pdf](https://cdn.realfood.gov/DGA_508.pdf) | Public domain (17 U.S.C. §105) | PDF, 9 content pages | ~4,500 |
| 2 | WHO "Healthy diet" fact sheet | [who.int/publications/m/item/healthy-diet-factsheet394](https://www.who.int/publications/m/item/healthy-diet-factsheet394) | CC BY-NC-SA 3.0 IGO (attribution, non-commercial) | HTML | ~1,200 |
| 3 | WHO "Healthy diet: keys to eating well" Q&A | [who.int/news-room/questions-and-answers/item/healthy-diet-keys-to-eating-well](https://www.who.int/news-room/questions-and-answers/item/healthy-diet-keys-to-eating-well) | CC BY-NC-SA 3.0 IGO | HTML | ~1,500 |
| 4 | NHS Eatwell Guide explainer | [nhs.uk/.../the-eatwell-guide](https://www.nhs.uk/live-well/eat-well/food-guidelines-and-food-labels/the-eatwell-guide/) | Crown copyright (verify OGL before commercial use) | HTML | ~1,800 |
| 5 | FNRI Nutritional Guidelines for Filipinos (2012) | [NGF-2012.pdf](https://www.fnri.dost.gov.ph/images/images/standardtools/NGF-2012.pdf) | Philippine gov't work, RA 8293 §176 (free non-commercial use; approval needed for profit use) | PDF | ~3,000 (estimate — not directly fetched; typical 10-message NGF documents run 15-25 pages) |

**Document count:** 5. **Total estimated tokens: ~12,000** (well within a single demo index).

**Estimation method:** For docs 1–4 the text was fetched directly (PDF read tool / WebFetch → markdown) and token count estimated at ~1.3 tokens per English word (standard GPT/Claude-family heuristic), counted against the actual word count of the extracted text. Doc 5 (FNRI) was not fetched byte-for-byte in this pass — its estimate is extrapolated from the known structure (10 guideline messages + rationale, similar density to comparable national FBDG documents) and should be confirmed by actually downloading `NGF-2012.pdf` before indexing.

**Why exclude EFSA DRVs and the raw Eatwell PDF from v1:**
- EFSA DRV opinions are CC BY-ND (no-derivatives) — chunking prose into a vector index is a live legal question there; including them either needs a fact-only representation (numeric DRVs, not prose spans) or EFSA sign-off. Not worth the risk for a demo.
- The UK Eatwell Guide's own PDF is one infographic page with almost no prose to chunk; the NHS explainer page (already included, item 4) carries the actual sentences and is fully sufficient to represent UK guidance for a demo.

**Licence-driven ground rules for the corpus, regardless of which docs are chosen:**
- Anything under CC BY-NC-SA (WHO) → demo/non-commercial use only; revisit before monetizing.
- Anything under CC BY-ND (EFSA Journal) → do not chunk prose verbatim into a shared index without legal review; prefer citing numeric values as facts.
- Anything under OGL / Crown copyright (UK) → safe for commercial use with attribution.
- Anything that's a US or PH government work (DGA, MyPlate, nutrition.gov, FNRI) → free to use; PH content additionally needs FNRI's prior approval only if the product is monetized.
- Store the licence + attribution string alongside every chunk's metadata at ingest time, not just in this doc — that's what makes "chunk N came from doc X under licence Y" auditable later.
