# Where should nutrient values for the local Filipino-dish table come from?

Research for issue #17. Follows from #15 (keep ~20 common Filipino dishes in a
local Postgres table instead of FDC/OFF). Pure research — no code changes.

Research date: 2026-08-01. All URLs below were fetched live on that date
unless marked as failed.

## 1. Is PhilFCT machine-readable (API/CSV) or web-lookup-only?

Short answer: **no official API or CSV/bulk-download product exists.** The
public-facing product is a login-gated web search tool. But in the course of
this research we found the tool's search-results page is, in practice,
reachable **without logging in** and returns the entire dataset embedded in
one HTML page — that is a real but undocumented/unofficial access path, not
a supported API, and could change or be blocked at any time.

Details:

- The official entry point is `https://i.fnri.dost.gov.ph/login/fct`, which
  is the login screen for "iFNRI", FNRI's portal of internal tools (PhilFCT
  is one of several: iAssess, iServe, iPromote, iBusiness, iLearn, iTrain).
  Fetched 2026-08-01, confirmed login-gated.
- The PhilFCT landing page `https://i.fnri.dost.gov.ph/fct/library`
  describes it as "the free online access of data on energy and nutrients of
  1600 foods commonly consumed in the country" and links to
  `https://i.fnri.dost.gov.ph/fct/library/starting_pg`, whose "Continue"
  button posts to a login form (`action="https://i.fnri.dost.gov.ph/login"`).
  Sub-pages linked from the nav (FAQ, History, Nutrient Information,
  Project Team) all bounced to the same login-required shell when fetched
  unauthenticated — no terms-of-use text was visible without an account.
- **Unofficial finding:** `https://i.fnri.dost.gov.ph/fct/library/search_item`
  returned, via a plain unauthenticated `curl`, a ~17 MB HTML page
  containing a `<tr>` row plus a hidden Bootstrap modal (full Proximates /
  Other Carbohydrate / Minerals / Vitamins / Lipids tables, "Amount per
  100 g E.P.") for every food in the database — 1,542 distinct food entries
  counted by unique modal IDs, consistent with the "~1500–1600 foods"
  figure quoted on FNRI's own pages. There is no `robots.txt` at
  `i.fnri.dost.gov.ph` (returns 404) restricting this.
- Individual per-food print reports are also fetchable directly by numeric
  ID with no auth, e.g. `https://i.fnri.dost.gov.ph/fct/library/report/4216`
  returned a 2-page PDF for "Egg, chicken, whole, boiled" (Food ID `H004`),
  watermarked "DOST-FNRI. Philippine Food Composition Table Online Database
  (PhilFCT), Release 1 December 2019" with a live `Report Date` timestamp
  from the moment of the request.
- This means: the underlying data is technically scrapable today, but (a)
  it's not a documented/stable interface, (b) FNRI could add auth
  enforcement to that endpoint without notice, and (c) using it at all
  raises the licensing question answered in §2 below.
- Two official FNRI seminar-series conference abstracts (their own
  self-published PDFs) confirm PhilFCT is a live, actively maintained
  project, not a defunct tool:
  - [PhilFCT Online Database: Data Updates, Features and Security
    Enhancement](https://www.fnri.dost.gov.ph/images/sources/SeminarSeries/45th/PhilFCT.pdf)
    (45th FNRI Seminar Series, 2019 abstract book) — describes ongoing
    updates and 142,710 recorded user hits as of December 2018.
  - [DOST-FNRI launches updated nutrition tools](https://fnri.dost.gov.ph/index.php/programs-and-projects/news-and-announcement/779-dost-fnri-launches-updated-nutrition-tools)
    — news post dated around the Feb 19, 2020 launch of the "Philippine
    Food Composition Tables 2019" print handbook and 4th-edition Food
    Exchange Lists; does not mention an API or bulk download, and says
    pricing/availability for the handbooks would be "posted via the
    DOST-FNRI official website and Social Media Pages" — i.e., the
    authoritative distribution channel for the print FCT is a paid physical
    or PDF handbook, not open data.
- A Freedom-of-Information request page,
  `https://www.foi.gov.ph/requests/food-composition-table/`, exists
  (indicating someone has formally requested FCT data via the PH FOI
  portal) but returned HTTP 403 Forbidden when fetched — could not confirm
  the outcome of that request. Noted as inaccessible, not silently dropped.

**Conclusion for Q1:** No supported API/CSV. Manual web-lookup is the
sanctioned path. An unauthenticated bulk HTML dump exists today at a
specific URL as an implementation detail of the search UI, not a product —
treat it as fragile and revisit the licensing question (§2) before relying
on it for anything beyond one-time verification lookups.

## 2. Licence / attribution

### What RA 8293 §176 actually says

Fetched directly from the primary source,
[lawphil.net — R.A. 8293, Intellectual Property Code of the Philippines](https://lawphil.net/statutes/repacts/ra1997/ra_8293_1997.html)
(verified by downloading the raw HTML and locating Section 176 verbatim):

> **Section 176.** *Works of the Government.* — 176.1. No copyright shall
> subsist in any work of the Government of the Philippines. However, prior
> approval of the government agency or office wherein the work is created
> shall be necessary for exploitation of such work for profit. Such agency
> or office may, among other things, impose as a condition the payment of
> royalties. No prior approval or conditions shall be required for the use
> of any purpose of statutes, rules and regulations, and speeches,
> lectures, sermons, addresses, and dissertations, pronounced, read or
> rendered in courts of justice, before administrative agencies, in
> deliberative assemblies and in meetings of public character. (Sec. 9,
> first par., P.D. No. 49)
>
> 176.2. The author of speeches, lectures, sermons, addresses, and
> dissertations mentioned in the preceding paragraphs shall have the
> exclusive right of making a collection of his works. (n)
>
> 176.3. Notwithstanding the foregoing provisions, the Government is not
> precluded from receiving and holding copyrights transferred to it by
> assignment, bequest or otherwise; nor shall publication or republication
> by the Government in a public document of any work in which copyright is
> subsisting be taken to cause any abridgment or annulment of the copyright
> or to authorize any use or appropriation of such work without the
> consent of the copyright owner. (Sec. 9, third par., P.D. No. 49)

This matches the ticket's framing exactly: PhilFCT, as a work of a
Philippine government agency (FNRI/DOST), carries **no copyright** for
general use, **but** "prior approval ... shall be necessary for
exploitation of such work for profit," and FNRI "may ... impose as a
condition the payment of royalties." The §176.1 second sentence's exception
(no approval needed for statutes/speeches/etc.) does **not** cover a food
composition dataset, so PhilFCT does not fall under that carve-out —
NutriGraph would fall under the general "prior approval for profit" clause,
not the exemption.

Secondary sources (Brainly, eCodal, eLibrary summaries) that surfaced
during search all quote this identically, which cross-checks the primary
text above.

### Does FNRI publish its own terms of use / attribution requirement?

Not found. Every FNRI/iFNRI page that would plausibly carry ToU or
attribution language (FAQ, History, Nutrient Information at
`i.fnri.dost.gov.ph/fct/library/{faq,history,nutriinfo}`) redirected to the
login-required shell when fetched without an account, so no explicit terms
text was retrievable in this research pass. The only usage guidance
publicly visible is the one-line description on the library landing page
calling PhilFCT "free online access" — that is marketing copy, not a
licence grant, and doesn't override RA 8293 §176.

### Does "commercial exploitation needs prior approval" apply to NutriGraph?

Yes, on a plain reading of §176.1: if NutriGraph is or could become a
commercial/monetized product, using FNRI-sourced nutrient figures is
"exploitation of such work for profit," so prior approval from
FNRI/DOST is required, and FNRI is entitled to impose royalty terms. The
statute does not define a de minimis or research exception. There's no
FNRI-published process for requesting this approval found in this
research pass (their "Contact Us" page,
`https://i.fnri.dost.gov.ph/contacts/contacts`, was linked but not
independently fetched/verified for a data-licensing contact form — noted
as unverified rather than assumed).

**Practical implication:** manually re-keying ~20 individual nutrient
values (numbers are not copyrightable per se in most jurisdictions, though
Philippine law doesn't spell out a "sweat of the brow" exception the way
some others do) is lower-risk than bulk-scraping/republishing FNRI's
dataset wholesale. Before shipping this feature commercially, get an
explicit answer from FNRI on whether citing ~20 hand-entered figures with
attribution requires their sign-off — don't assume silence means yes.

## 3. Field alignment: PhilFCT vs. FDC

Confirmed directly from data fetched today (see §1, the per-food modal
data), a PhilFCT entry provides, per 100 g edible portion:

- **Proximates:** Water (g), Energy (kcal, calculated), Protein (g), Total
  Fat (g), Carbohydrate total (g), Ash total (g)
- **Other Carbohydrate:** Fiber, total dietary (g); Sugars, total (g)
- **Minerals:** Calcium, Phosphorus, Iron, (sometimes) Potassium, Sodium,
  (sometimes) Zinc
- **Vitamins:** Retinol/Vitamin A, beta-Carotene, RAE, Thiamin (B1),
  Riboflavin (B2), Niacin, (sometimes) Ascorbic Acid (C)
- **Lipids:** saturated/monounsaturated/polyunsaturated fatty acids,
  Cholesterol (for animal-derived foods)

This is a **superset** of what NutriGraph likely needs (kcal, protein,
fat, carb, fiber, sugar, sodium) plus useful extras (cholesterol, fatty
acid breakdown, vitamin A/B1/B2/C). Field names map cleanly to FDC's
nutrient set conceptually (same units, g/mg/µg per 100g), though FDC uses
numeric nutrient IDs and PhilFCT uses a fixed report layout — any import
would need manual field-mapping, not an automated schema match.

**Confirmed gap, not assumed:** the ticket asked us to verify rather than
assume that PhilFCT may not report sugar/sodium for every item. This is
true, and worse than "may not" — it's the documented history. Per FNRI's
own conference abstract,
[Updating of the Philippine FCT Using Indirect Method — Phase 1: TDF, Total
Sugars, Sodium, Available Carbohydrate and Energy](https://fnri.dost.gov.ph/images/sources/SeminarSeries/43rd/Updating-of-FCT.pdf)
(43rd FNRI Seminar Series abstract):

> "In the country, the Philippine Food Composition Tables (FCT) 1997 is the
> current publication of the nutrient data of foods. **However, this has no
> data on nutrients with health implications, such as total dietary fiber
> (TDF), sugar and sodium (Na).**"

The same abstract states the fix was retrofitted via an *indirect* method —
"TDF, total sugars and Na data were borrowed from foreign databases, and
adapted to the water content of Philippine FCT food items" (i.e., not
locally chemically analyzed for most items) — and only for a subset of the
database: **1,246 items got TDF, 1,163 got total sugars, 1,298 got sodium**,
against a database of ~1,500–1,600 total items. So a meaningful minority of
entries still lack sugar/sodium, and where present, those specific fields
may be borrowed/estimated rather than directly measured for the Philippine
food item. This was directly observed in our own live data pull too: e.g.
"Tapa, baboy" (F243) and "Longanisa, baboy" (F257) both show `Sugars, total
(g) = -` (no data), while other fields are populated.

## 4. Reference amount: per-100g or per-serving?

**Per 100 g edible portion (E.P.)** — confirmed directly, every nutrient
panel fetched today is headed "Amount per 100 g E.P." This matches standard
FCT convention and FDC's per-100g convention, so no conversion is needed
structurally — only edible-portion-percentage matters (PhilFCT also states
an "Edible portion" percentage per food, e.g. 89% for boiled egg, 70% for
roast chicken, 100% for jerky/rice items — this is the trimmed/bone-removed
fraction, not a serving weight).

**Serving weights for common dishes:** FNRI/DOH/WHO/NNC jointly publish the
["Pinggang Pinoy"](https://www.fnri.dost.gov.ph/index.php/116-pinggang-pinoy)
food-plate guide, which gives per-meal food-group proportions (a plate
model, not a nutrient table) rather than a table of gram weights per dish.
The FNRI page itself, fetched today, describes it only as "a new,
easy-to-understand food guide that uses a familiar food plate model to
convey the right food group proportions on a per-meal basis" — the actual
gram/serving figures live in a linked PDF handout that this pass did not
extract text from (the page references a downloadable guide; content was
not retrieved as readable text). Do not treat any specific gram number as
sourced from this pass unless it's fetched and quoted directly — flagging
this as an open follow-up rather than fabricating numbers, per the "don't
guess" instruction for this ticket.

## 5. Other credible sources

- **ASEANFOODS / ASEAN Food Composition Database.** ASEANFOODS (Association
  of Southeast Asian Networks of Food Data Systems), regional secretariat
  at the Institute of Nutrition, Mahidol University (INMU), Thailand,
  published the ASEAN Food Composition Tables (2000), combining data from
  six countries including the Philippines — "Dr Aida Aguinaldo of the
  Philippines Food and Nutrition Institute" is named as a technical
  committee member, and "the Philippine Food Composition Tables retained 17
  of the food groupings based on the ASEAN FCT Major Food Groupings" per
  [INMU's ASEANFOODS composition-data
  page](https://inmu2.mahidol.ac.th/aseanfoods/composition_data.html).
  A Philippines-specific country report PDF exists at
  `https://inmu.mahidol.ac.th/aseanfoods/doc/07_Country%20Report%20(Phillippines).pdf`
  but this pass could not extract readable text from it (returned as raw
  PDF binary the fetch tool couldn't parse) — noted as inaccessible rather
  than summarized from assumption. ASEANFOODS is effectively downstream of
  PhilFCT (same source institute), so it's not an independent primary
  source for dish-level data, just a regional aggregation layer.
- **Peer-reviewed Philippine nutrition research on cooked dishes.** Not
  independently verified in this pass beyond the FNRI Seminar Series
  abstracts already cited (those are FNRI's own institutional conference
  proceedings, not a third-party journal, but they are the closest thing
  found to "primary literature on FCT methodology"). No Philippine Journal
  of Science article specifically analyzing composition of cooked
  ulam-style dishes (adobo, sinigang, etc.) was located and fetched in this
  pass — this remains an open lead, not a dead end, for a future research
  pass with more search budget.
- **Non-primary fallback (mention only, do not cite for numbers):**
  MyFitnessPal, food blogs, and similar unsourced calculators surface
  heavily in generic search results for "calories in adobo" style queries
  and were deliberately excluded from this document per the ticket's
  instruction — they are not used anywhere above.

## 6. Candidate list of ~20 dishes

Every PhilFCT figure below was pulled from the live, unauthenticated
dataset at `https://i.fnri.dost.gov.ph/fct/library/search_item` (fetched
2026-08-01; see §1 for why this URL is reachable and its caveats) and
cross-checked against the same food's individual PDF report where
convenient. The **PhilFCT Food ID** is FNRI's own code for that entry —
use it to re-locate the exact record for verification or if the site
changes. All values are **per 100 g edible portion**.

Important, and expected per the ticket's own framing: **PhilFCT does not
contain most of these dishes in their everyday home-cooked form.** Where a
real number is shown below, check the "Note" column — several are
**canned/commercial or jerky/raw variants**, not the fresh home-cooked
ulam a Filipino household would actually make. Six of the twenty have no
PhilFCT entry at all under any name we searched.

### Sourced from PhilFCT (real numbers, fetched, not invented)

| Dish (candidate name) | PhilFCT English name / Food ID | kcal | Protein (g) | Fat (g) | Carb (g) | Fiber (g) | Sugar (g) | Sodium (mg) | Note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Sinangag (garlic fried rice) | Rice, well-milled, fried — `A021` | 187 | 2.6 | 4.0 | 35.2 | 0.5 | 0.1 | 4 | Good match — this is the actual dish. |
| Pandesal | Bread, pan de sal — `A042` | 330 | 10.1 | 4.2 | 62.9 | 2.5 | 7.6 | 596 | Good match. |
| Arroz caldo | Rice gruel w/ chicken — `R011` | 63 | 2.0 | 0.4 | 12.8 | 0.5 | 0.7 | 453 | Good match. |
| Champorado | Rice gruel w/ choc & milk — `R012` | 97 | 1.7 | 0.2 | 22.0 | 0.7 | 4.8 | 13 | Good match. |
| Lechon (whole roast, proxy) | Chicken, whole, seasoned, roasted ("Lechon manok") — `F265` | 226 | 24.8 | 12.0 | 4.7 | 0.0 | 1.5 | 716 | Roast **chicken**, not the classic whole roast pig — different dish, same cooking style. |
| Tapa (pork, dried/cured) | Pork jerky ("Tapa, baboy") — `F243` | 277 | 15.8 | 19.2 | 10.2 | 0.0 | no data | 1086 | Dried/jerky form, not pan-fried tapsilog-style tapa. |
| Tapa (beef, dried/cured) | Beef jerky ("Tapa, baka") — `F215` | 102 | 13.8 | 3.6 | 3.7 | 0.0 | 1.2 | 694 | Same caveat as above. |
| Longganisa (pork sausage) | Sausage, pork ("Longanisa, baboy") — `F257` | 315 | 11.8 | 22.7 | 15.8 | 0.9 | no data | 714 | Reasonable match, raw/cured sausage. |
| Adobo (pork) | Pork adobo, canned — `R050` | 277 | 10.3 | 24.8 | 3.1 | 0.3 | 0.1 | 254 | **Canned commercial product, not home-cooked adobo.** |
| Caldereta (beef) | Beef caldereta, canned — `R027` | 284 | 13.3 | 24.9 | 1.6 | 1.1 | no data | 315 | **Canned, not home-cooked.** |
| Kare-kare (beef) | Beef kare-kare, canned — `R031` | 64 | 9.0 | 1.1 | 4.4 | 1.5 | 1.7 | 109 | **Canned, not home-cooked** (no peanut-sauce richness — numbers look implausibly lean for a home version). |
| Dinuguan (pork) | Pork blood stew, canned ("Dinuguan baboy, de lata") — `R053` | 155 | 14.8 | 10.6 | 0.0 | 0.2 | 0.0 | no data | **Canned, not home-cooked.** |
| Pancit (as pancit molo) | Wonton soup, prepared, w/ meat-like protein ("Pancit molo, w/ MLP") — `R100` | 93 | 5.0 | 5.3 | 6.3 | 3.6 | 2.8 | 320 | Only pancit variant found; not pancit canton/bihon/palabok. |
| Lumpia (Shanghai, raw filling) | Spring roll, Shanghai, uncooked ("Lumpia, Shanghai, hindi luto") — `R064` | 266 | 5.8 | 10.0 | 38.3 | 1.2 | 0.7 | 275 | **Uncooked filling+wrapper mix, not the fried finished lumpia.** |

### Not found in PhilFCT — manual lookup required

Searched by English and Filipino name (and common variants/spellings)
against the full live dataset; zero matches for all six:

- **Sinigang (na baboy)** — sour tamarind soup
- **Tinola** — chicken ginger-and-vegetable soup
- **Bulalo** — beef shank/bone-marrow soup
- **Pinakbet** — mixed-vegetable stew with bagoong
- **Sisig** — sizzling chopped pork
- **Chicken inasal** — Bacolod-style grilled chicken

(Also checked and absent, not part of the final 20 but worth knowing:
laing, bicol express, menudo, halo-halo, pancit canton/bihon, lechon
kawali, lechon baboy — same "not in PhilFCT" verdict.)

For each of these, the exact manual steps a human would take:

1. Go to `https://i.fnri.dost.gov.ph/login/fct` and register a free account
   at `https://i.fnri.dost.gov.ph/users/register` (registration was not
   completed in this research pass — we did not create test credentials).
2. Log in, open the PhilFCT search tool (`.../fct/library/starting_pg`
   after login), and use the typeahead search box (the page loads
   `bootstrap-typeahead.js`, confirming a name-based autocomplete search
   exists once authenticated).
3. Search the Filipino dish name (e.g. "sinigang", "tinola", "bulalo",
   "pinakbet", "sisig", "inasal") and any obvious English gloss ("sour
   soup", "grilled chicken").
4. If a matching combination/mixed dish exists, open it and read the
   Proximates tab (Energy, Protein, Fat, Carbohydrate), Other Carbohydrate
   tab (Fiber, Sugars), and Minerals tab (Sodium) — same layout documented
   in §3/§6 above.
5. If no matching combination dish exists (likely, since none of these six
   turned up in the full unauthenticated dataset dump either), the
   practical alternative is: pull PhilFCT raw-ingredient values for each
   component (e.g., for sinigang: pork, tamarind, kangkong, radish, string
   beans, taro) and do the recipe-calculation yourself — FNRI's own
   updating methodology (§3) uses exactly this "indirect method" internally
   (INFOODS Compilation Tool + recipe calculation module) when it lacks a
   direct match, which is a validated approach, not an improvisation.
6. Alternative fetch: try
   `https://i.fnri.dost.gov.ph/fct/library/search_item` unauthenticated
   (works as of 2026-08-01, see §1) and text-search the page for the dish
   name before assuming a login is required — this is how the 14 sourced
   rows above were actually found.

## Confidence / freshness note

- Today's date for this research: **2026-08-01**.
- PhilFCT's underlying data release is watermarked "Release 1 December
  2019" on every generated report — the dataset itself is **over 6 years
  old** as of this research and there is no evidence in this pass of a
  newer release. Treat any PhilFCT figure as "best available, not
  current-year-verified."
- The unauthenticated bulk-access path documented in §1 is an
  **implementation detail observed on one date**, not a guaranteed stable
  interface. Re-verify it works before building any tooling that depends
  on it, and don't assume it will still work by the time this ticket is
  implemented.
- Two fetches failed outright and are flagged rather than silently
  dropped: `https://www.foi.gov.ph/requests/food-composition-table/` (HTTP
  403) and the ASEANFOODS Philippines country-report PDF (fetched but
  unreadable as binary in this pass).
- Pinggang Pinoy gram-level serving sizes were **not** confirmed with a
  directly-fetched, quotable source in this pass — do not treat any
  specific serving-gram number for Pinggang Pinoy as sourced by this
  document.

## Recommendation

1. **Do not build an automated importer against PhilFCT.** There's no
   supported API/CSV, the dataset is 6+ years stale, and the one
   unauthenticated bulk-access path found here is undocumented and could
   disappear.
2. **Manual, one-time data entry** for the ~20-dish local table is the
   right approach, exactly as the ticket anticipated. For the 14 dishes
   above with a PhilFCT figure, use that figure but **relabel/flag which
   are canned or otherwise non-home-cooked proxies** (adobo, caldereta,
   kare-kare, dinuguan, pancit molo, lumpia Shanghai raw) so the app
   doesn't present a canned-food number as "homemade adobo." For the 6 with
   no PhilFCT entry, either do the recipe-calculation-from-ingredients
   approach (§6 step 5) or accept a wider error bar and clearly source it
   as "estimated from raw-ingredient composition, not FNRI-verified."
3. **Record the PhilFCT Food ID** (e.g. `A021`, `R011`) alongside every
   number entered into the local table, so a future maintainer can
   re-verify against the live tool without re-doing this research.
4. **Resolve the RA 8293 §176 licensing question before shipping
   commercially.** If NutriGraph is or becomes a paid/commercial product,
   confirm with FNRI whether citing ~20 individually re-keyed figures (not
   a bulk republish of their dataset) needs their prior approval under
   §176.1, and get that answer in writing rather than assuming "government
   work = free to use commercially."
5. **Revisit if FNRI ever ships a real API/CSV** — the 2019/2020 "updated
   nutrition tools" launch shows FNRI does periodically modernize this
   product; check `fnri.dost.gov.ph` news announcements periodically rather
   than assuming the current login-gated, stale-dataset state is permanent.
