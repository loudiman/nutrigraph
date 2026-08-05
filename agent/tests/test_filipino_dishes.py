"""The Filipino dish seed. The checks a wrong nutrient value would still pass, and
the one it would not: every calculated row is recomputed from its own components.

No network. The file is the fixture."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

SEED = json.loads(
    (Path(__file__).resolve().parents[1] / "seeds" / "filipino_dishes.json").read_text(encoding="utf-8")
)
AUDIT = json.loads(
    (Path(__file__).resolve().parents[1] / "seeds" / "filipino_dishes_audit.json").read_text(encoding="utf-8")
)
DISHES = SEED["dishes"]
COLUMNS = ("kcal", "protein_g", "fat_g", "carb_g", "fibre_g", "sodium_mg")
WHOLE = ("kcal", "sodium_mg")


# --- shape --------------------------------------------------------------------


def test_about_twenty_dishes():
    assert 18 <= len(DISHES) <= 22


def test_every_row_carries_the_columns_a_diet_filter_needs():
    for d in DISHES:
        assert d["name"] and d["aliases"] and d["tags"]
        assert d["value_kind"] in ("direct", "proxy", "calculated")
        for c in COLUMNS:
            assert c in d


# --- provenance ---------------------------------------------------------------


def test_direct_and_proxy_rows_cite_a_philfct_food_id():
    for d in DISHES:
        if d["value_kind"] in ("direct", "proxy"):
            assert d["philfct_food_id"] in AUDIT["foods"], d["name"]


def test_proxy_rows_say_what_the_proxy_actually_is():
    for d in DISHES:
        if d["value_kind"] == "proxy":
            assert d["source_note"].startswith("PROXY:"), d["name"]


def test_calculated_rows_record_components_grams_and_arithmetic():
    for d in DISHES:
        if d["value_kind"] != "calculated":
            continue
        calc = d["calculation"]
        assert calc["components"] and calc["method"] and calc["assumption"]
        assert calc["recipe_weights_sourced"] is False
        for c in calc["components"]:
            assert c["philfct_food_id"] in AUDIT["foods"]
            assert c["grams"] > 0


def test_the_direct_and_proxy_columns_match_the_audit_file_verbatim():
    keys = {
        "kcal": ("proximates", "Energy, calculated (kcal)"),
        "protein_g": ("proximates", "Protein (g)"),
        "fat_g": ("proximates", "Total Fat (g)"),
        "carb_g": ("proximates", "Carbohydrate, total (g)"),
        "fibre_g": ("other_carbohydrate", "Fiber, total dietary (g)"),
        "sodium_mg": ("minerals", "Sodium, Na (mg)"),
    }
    for d in DISHES:
        if d["value_kind"] == "calculated":
            continue
        panels = AUDIT["foods"][d["philfct_food_id"]]["panels"]
        for col, (panel, label) in keys.items():
            raw = panels.get(panel, {}).get(label, "-")
            assert d[col] == (None if raw == "-" else float(raw)), (d["name"], col)


# --- the check a wrong number would not survive -------------------------------


def test_every_calculated_row_recomputes_from_its_own_components():
    for d in DISHES:
        if d["value_kind"] != "calculated":
            continue
        calc = d["calculation"]
        total = Decimal(calc["total_dish_weight_g"])
        assert total == sum(Decimal(c["grams"]) for c in calc["components"]) + Decimal(calc["water_added_g"])
        for col in COLUMNS:
            raws = [c["per_100g"][col] for c in calc["components"]]
            if any(r == "-" for r in raws):
                assert d[col] is None, (d["name"], col)  # never zero, never estimated
                continue
            got = sum(Decimal(r) * Decimal(c["grams"]) / 100 for r, c in zip(raws, calc["components"]))
            per100 = got * 100 / total
            want = int(per100.to_integral_value()) if col in WHOLE else float(round(per100, 1))
            assert d[col] == want, (d["name"], col)


def test_a_missing_source_value_is_null_and_never_zero():
    # PhilFCT prints '-' for dinuguan's sodium; the row must carry that through.
    dinuguan = next(d for d in DISHES if d["name"] == "Dinuguan")
    assert dinuguan["sodium_mg"] is None
