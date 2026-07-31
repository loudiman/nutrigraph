# Food composition database for logged-meal nutrient lookup

Research for issue #5: which database gives nutrient values for a logged meal.
Date: 2026-08-01. Primary sources only, cited per claim.

## Candidates compared

1. USDA FoodData Central (FDC)
2. Open Food Facts (OFF)
3. Edamam Food Database API (added — it's the layer that answers the free-text
   matching requirement neither FDC nor OFF solve on their own)

## Licence & attribution

- **FDC**: public domain, CC0 1.0 Universal. No permission needed; USDA
  requests (not requires) that FoodData Central be listed as the source.
  [fdc.nal.usda.gov/help](http://fdc.nal.usda.gov/help/)
- **OFF**: Open Database License (ODbL) for data, CC-BY-SA for images.
  ODbL is share-alike — redistributing a derived database requires
  attribution and keeping the derivative open under ODbL.
  [world.openfoodfacts.org/data](https://world.openfoodfacts.org/data)
- **Edamam**: proprietary API, commercial terms (paid tiers up to
  $999/mo, free tier below). No open data licence — you're licensing API
  access, not redistributable data.
  [developer.edamam.com/food-database-api-docs](https://developer.edamam.com/food-database-api-docs)

## Coverage

- **FDC**: Foundation Foods and SR Legacy are US-focused generic foods
  (per-100g). FNDDS is US NHANES survey data (US dietary patterns only).
  Branded Foods is the "USDA Global Branded Foods Products Database" —
  name implies international brands, but the help page does not detail
  non-US coverage explicitly. [fdc.nal.usda.gov/help](http://fdc.nal.usda.gov/help/), [fdc.nal.usda.gov/download-datasets](http://fdc.nal.usda.gov/download-datasets/)
- **OFF**: crowd-sourced globally, 1.65M+ products across 180+ countries
  per Open Food Facts' own product-count tracker — strongest for branded/
  packaged foods, including foods likely to be barcode-scanned in the
  Philippines. No generic/unbranded "raw egg" type entries by default —
  it's a barcode-product database. [world.openproductsfacts.org/product-count](https://world.openproductsfacts.org/product-count)
- **Edamam**: indexes generic (non-branded), packaged/branded, fast-food,
  and generic multi-ingredient meals. Coverage skews US/EU; no explicit
  Philippines coverage claim in docs.
  [developer.edamam.com/food-database-api-docs](https://developer.edamam.com/food-database-api-docs)

**Implication for a PH-based demo user**: neither FDC's generic datasets
nor Edamam's index guarantee Philippine dishes (e.g. adobo, pandesal).
OFF's barcode coverage is the best bet for branded PH packaged goods but
is weak/absent for generic home-cooked dishes.

## Access: API vs bulk, rate limits, keys

- **FDC API**: `https://api.nal.usda.gov/fdc/v1/`. Requires a data.gov
  API key on every request. Default limit 1,000 req/hour/IP; DEMO_KEY is
  throttled to 30/hour, 50/day. 429 on excess, with `X-RateLimit-*`
  headers. Endpoints: `/food/{fdcId}`, `/foods`, `/foods/list`,
  `/foods/search`. [fdc.nal.usda.gov/api-guide](http://fdc.nal.usda.gov/api-guide/)
- **FDC bulk**: CSV or JSON, per data type or "Full Download of All Data
  Types." Zipped 195MB-460MB, unzipped 3.1GB-3.7GB per type (Branded is
  largest). Foundation/Branded updated quarterly, FNDDS biennially, SR
  Legacy frozen. [fdc.nal.usda.gov/download-datasets](http://fdc.nal.usda.gov/download-datasets/)
- **OFF API**: `https://world.openfoodfacts.org`, no API key for reads,
  but a custom User-Agent is required. Rate limits: 15 req/min/IP for
  product queries, 10 req/min/IP for search queries — explicitly too low
  for search-as-you-type. IP bans on abuse.
  [openfoodfacts.github.io/openfoodfacts-server/api](https://openfoodfacts.github.io/openfoodfacts-server/api/)
- **OFF bulk**: nightly CSV (~0.9GB gzip / ~9GB uncompressed), MongoDB
  dump, JSONL, Parquet (via Hugging Face). Delta exports for last 14
  days. [world.openfoodfacts.org/data](https://world.openfoodfacts.org/data)
- **Edamam**: API-key (app_id) required. Free tier: 1,000 req/day, 50
  req/min. Paid scales up. No bulk export — API access only.
  [developer.edamam.com/food-database-api-docs](https://developer.edamam.com/food-database-api-docs)

## Data shape: portions, nutrients, units

- **FDC**: `foodPortions[]` gives household measures, e.g.
  `{"amount":1,"modifier":"medium","gramWeight":118}`. `foodNutrients[]`
  gives `{nutrient:{id,name,unitName}, amount}` — amount is per 100g for
  Foundation/SR Legacy; Branded Foods instead carry `servingSize` +
  `servingSizeUnit` on the record. Units are standard (g, mg, mcg, kcal,
  kJ). [fdc.nal.usda.gov/api-spec/fdc_api.html](https://fdc.nal.usda.gov/api-spec/fdc_api.html)
- **OFF**: `nutriments` object with paired fields per nutrient — `_100g`
  suffix (value per 100g/100ml) and `_serving` suffix (value per the
  product's declared `serving_size`, itself in grams). Nutrient set is
  whatever the product label reports — sparser and less standardized
  than FDC. [openfoodfacts.github.io/openfoodfacts-server/api](https://openfoodfacts.github.io/openfoodfacts-server/api/)
- **Edamam**: returns 28+ nutrients (macros + micros + allergens) plus a
  list of valid measures (household units, qualifiers like "large"/
  "small") per matched food, so portion resolution is closer to how a
  logged meal is entered by a user. [developer.edamam.com/food-database-api-docs](https://developer.edamam.com/food-database-api-docs)

## Search quality for free text ("two eggs and toast")

- **FDC** `/foods/search` is keyword search; no documented fuzzy or NLP
  matching, no query syntax beyond filters like `dataType`. A phrase like
  "two eggs and toast" would need to be split into food terms by the
  caller before hitting the endpoint. [fdc.nal.usda.gov/api-guide](http://fdc.nal.usda.gov/api-guide/)
- **OFF** does not do full-text search in the core Product Opener API at
  all — v2/v3 search is structured/filter-based (category, brand,
  nutrient). Full-text needs the separate Search-a-licious service, and
  even that isn't natural-language meal parsing.
  [openfoodfacts.github.io/openfoodfacts-server/api](https://openfoodfacts.github.io/openfoodfacts-server/api/)
- **Edamam** is the one built for this: its "built-in food-logging
  context" explicitly supports NLP requests for "chatbots and natural
  language calorie counters" — i.e., it can take a sentence like "two
  eggs and toast" and split/match it to foods + quantities.
  [developer.edamam.com/food-database-api-docs](https://developer.edamam.com/food-database-api-docs)

None of the three raw databases eliminate the need for an app-side
matching/parsing layer unless Edamam's NLP endpoint is used directly.

## Recommendation

**Primary: USDA FoodData Central.** Public-domain (CC0) data with no
redistribution obligations, generous free rate limit (1,000/hr with a
free key), and the deepest/most standardized nutrient fields (per-100g
generic foods plus branded). Best fit for a free NutriGraph project
where licence risk and cost must stay at zero.

**Fallback: yes, Open Food Facts is needed**, specifically for branded/
packaged goods a PH-based user scans or logs by brand name that FDC's
Branded Foods dataset won't have (non-US products). Use OFF as a
secondary lookup keyed by barcode/brand match, not as the primary
nutrient source — its nutrient set is sparser and its data is
share-alike (ODbL), so track that attribution obligation separately from
FDC's.

**Free-text parsing is a separate concern from data source.** Neither
FDC nor OFF does NLP meal-sentence parsing. Either build a small
in-app matching layer (split sentence → look up terms in FDC/OFF) or
adopt Edamam's NLP endpoint as the entry point, using FDC as the
system of record for the nutrient values it returns. Given issue #5
only asks for the *database*, this is flagged as a follow-up decision,
not resolved here.

## Sources

- USDA FoodData Central API guide — http://fdc.nal.usda.gov/api-guide/
- USDA FoodData Central bulk downloads — http://fdc.nal.usda.gov/download-datasets/
- USDA FoodData Central help/attribution — http://fdc.nal.usda.gov/help/
- USDA FoodData Central OpenAPI spec — https://fdc.nal.usda.gov/api-spec/fdc_api.html
- Open Food Facts API docs — https://openfoodfacts.github.io/openfoodfacts-server/api/
- Open Food Facts data/licence page — https://world.openfoodfacts.org/data
- Open Food Facts product count — https://world.openproductsfacts.org/product-count
- Edamam Food Database API docs — https://developer.edamam.com/food-database-api-docs
