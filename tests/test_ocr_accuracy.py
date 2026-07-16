"""OCR accuracy tests: compare extract_stats() output against verified ground truth."""

import os
from pathlib import Path

import pytest

from ttr_tracker.database import Database, PartialRunRow
from ttr_tracker.ocr import Conf, extract_stats

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.environ.get("TTR_DB_PATH", str(BASE_DIR / "data" / "tracker.db"))
_db_instance: Database | None = None


def _get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(DB_PATH)
    return _db_instance

# Tolerance table: (high_conf_tol, medium_conf_tol)
# Medium confidence gets wider tolerance
FIELD_TOLERANCES: dict[str, tuple[float, float]] = {
    "score": (0, 0),
    "pieces_placed": (0, 0),
    "pps": (0.05, 0.15),
    "inputs": (0, 0),
    "kpp": (0.05, 0.15),
    "spp": (1.0, 3.0),
    "all_clears": (0, 0),
    "time_left": (1.0, 3.0),
}

STAT_TO_DB_ATTR: dict[str, str] = {
    "score": "score",
    "pieces_placed": "pieces_placed",
    "pps": "pps",
    "inputs": "inputs",
    "kpp": "kpp",
    "spp": "spp",
    "all_clears": "all_clears",
    "time_left": "time_left",
}

# Error categories for failure classification
CAT_VALUE_MISSING = "value_missing"
CAT_TOLERANCE = "tolerance"

# Per-process OCR cache: maps run_id -> extract_stats() result
# Avoids recomputing OCR across tests within the same process (critical for
# test_accuracy_summary which iterates all runs, and for xdist workers).
_OCR_CACHE: dict[int, dict] = {}


def _get_verified_runs() -> list[PartialRunRow]:
    with _get_db().session() as sess:
        return (
            sess.query(PartialRunRow)
            .filter(PartialRunRow.source_image.isnot(None))
            .order_by(PartialRunRow.id)
            .all()
        )


def _image_exists(run: PartialRunRow) -> bool:
    return run.source_image is not None and Path(run.source_image).is_file()


def _get_ocr_cached(run_id: int, image_path: str) -> dict:
    if run_id not in _OCR_CACHE:
        _OCR_CACHE[run_id] = extract_stats(image_path)
    return _OCR_CACHE[run_id]


# ── Fixtures ──


@pytest.fixture(scope="session")
def verified_runs():
    return _get_verified_runs()


# ── Dynamic parametrization ──


def pytest_generate_tests(metafunc):
    if "run_data" in metafunc.fixturenames:
        pairs = []
        ids = []
        for r in _get_verified_runs():
            if _image_exists(r):
                run_data = {
                    "id": r.id,
                    "score": r.score,
                    "pieces_placed": r.pieces_placed,
                    "pps": r.pps,
                    "inputs": r.inputs,
                    "kpp": r.kpp,
                    "spp": r.spp,
                    "all_clears": r.all_clears,
                    "time_left": r.time_left,
                }
                # Use image_path, not OCR result — lazily computed per-process
                pairs.append((run_data, r.source_image))
                ids.append(f"id={r.id}")
        metafunc.parametrize("run_data", pairs, ids=ids)


# ── Field comparison helper (single field, confidence-aware) ──


def _check_field(
    stat_field: str,
    ground_truth: float | int | None,
    ocr_entry: dict,
    tol_strict: float,
    tol_lenient: float,
) -> tuple[bool, str | None, str | None]:
    """Compare one OCR field against ground truth.
    
    Returns (passed, error_message, category).
    Skips fields where OCR confidence is 'missing' (the pipeline knew it failed).
    """
    ocr_val = ocr_entry.get("value")
    ocr_conf = ocr_entry.get("conf", Conf.missing)

    # --- Confidence "missing": the pipeline knew it couldn't find this field ---
    if ocr_conf == Conf.missing:
        if ground_truth is not None:
            # all_clears: None and 0 are equivalent
            if stat_field == "all_clears" and ground_truth == 0:
                return True, None, None
            return False, f"OCR=missing, expected={ground_truth}", CAT_VALUE_MISSING
        return True, None, None

    tol = tol_strict if ocr_conf == Conf.high else tol_lenient

    # --- Null handling ---
    if ocr_val is None and ground_truth is None:
        return True, None, None
    if stat_field == "all_clears":
        if (ocr_val is None or ocr_val == 0) and (ground_truth is None or ground_truth == 0):
            return True, None, None
    if ocr_val is None:
        return False, f"OCR=None, expected={ground_truth}", CAT_VALUE_MISSING
    if ground_truth is None:
        return False, f"OCR={ocr_val}, expected=None", CAT_TOLERANCE

    # --- Numeric comparison ---
    diff = abs(float(ocr_val) - float(ground_truth))
    if diff > tol:
        return (
            False,
            f"OCR={ocr_val}, expected={ground_truth}, "
            f"diff={diff:.2f} (tol={tol}, conf={ocr_conf})",
            CAT_TOLERANCE,
        )
    return True, None, None


# ── Test: per-field accuracy (confidence-aware) ──


@pytest.mark.ocr
def test_ocr_fields(run_data):
    """Compare each OCR field against ground truth, respecting confidence levels."""
    run, image_path = run_data
    ocr_result = _get_ocr_cached(run["id"], image_path)
    stats = ocr_result.get("stats", {})
    failures = []
    categories = []

    for stat_field, db_attr in STAT_TO_DB_ATTR.items():
        tol_strict, tol_lenient = FIELD_TOLERANCES[stat_field]
        ground_truth = run[db_attr]
        ocr_entry = stats.get(stat_field, {})
        passed, msg, cat = _check_field(
            stat_field, ground_truth, ocr_entry, tol_strict, tol_lenient
        )
        if not passed:
            failures.append(f"{stat_field}: {msg}")
            if cat:
                categories.append(cat)

    if failures:
        cat_summary = ""
        if categories:
            counts = {}
            for c in categories:
                counts[c] = counts.get(c, 0) + 1
            cat_summary = " [" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + "]"
        pytest.fail(f"[id={run['id']} score={run['score']}]{cat_summary} " + "; ".join(failures))


# ── Test: mathematical cross-validation ──


@pytest.mark.ocr
def test_cross_validation(run_data):
    """Verify mathematical consistency of OCR results."""
    run, image_path = run_data
    ocr_result = _get_ocr_cached(run["id"], image_path)
    stats = ocr_result.get("stats", {})
    failures = []

    def v(field: str) -> float | None:
        e = stats.get(field)
        return float(e["value"]) if e and e.get("value") is not None else None

    score = v("score")
    pieces = v("pieces_placed")
    pps = v("pps")
    inputs_val = v("inputs")
    kpp = v("kpp")
    spp = v("spp")
    time_left = v("time_left")

    # score ≈ spp * pieces
    if score is not None and spp is not None and pieces is not None and pieces > 0:
        expected = spp * pieces
        ratio = score / expected
        if ratio < 0.8 or ratio > 1.25:
            failures.append(
                f"score={score:.0f} != spp*pieces={spp}*{pieces}={expected:.0f} "
                f"(ratio={ratio:.2f})"
            )

    # inputs ≈ kpp * pieces
    if inputs_val is not None and kpp is not None and pieces is not None and pieces > 0:
        expected = kpp * pieces
        ratio = inputs_val / expected
        if ratio < 0.8 or ratio > 1.25:
            failures.append(
                f"inputs={inputs_val:.0f} != kpp*pieces={kpp}*{pieces}={expected:.0f} "
                f"(ratio={ratio:.2f})"
            )

    # pieces ≈ pps * elapsed (when time_left is available)
    if pieces is not None and pps is not None and time_left is not None:
        elapsed = 120.0 - time_left
        if elapsed > 0:
            expected = pps * elapsed
            ratio = pieces / expected
            if ratio < 0.8 or ratio > 1.25:
                failures.append(
                    f"pieces={pieces} != pps*elapsed={pps}*{elapsed:.1f}={expected:.1f} "
                    f"(ratio={ratio:.2f})"
                )

    if failures:
        pytest.fail(f"[id={run['id']} score={run['score']}] " + "; ".join(failures))


# ── Sanity checks ──


@pytest.mark.ocr
def test_all_images_have_source_image(verified_runs):
    missing = [r for r in verified_runs if not _image_exists(r)]
    if missing:
        msg = ", ".join(f"id={r.id} path={r.source_image}" for r in missing)
        pytest.fail(f"Images not found on disk: {msg}")


@pytest.mark.ocr
def test_ocr_available():
    from ttr_tracker.ocr import is_available

    assert is_available(), "Tesseract OCR is not available"


# ── Summary ──


@pytest.mark.ocr
def test_accuracy_summary():
    """Print a per-field accuracy table with error categories (uses cached OCR results)."""
    runs = _get_verified_runs()
    total = len(runs)
    if total == 0:
        pytest.skip("No verified runs with source images found")

    passed: dict[str, int] = {f: 0 for f in FIELD_TOLERANCES}
    errors: dict[str, list[tuple[int, float | int, float | int]]] = {
        f: [] for f in FIELD_TOLERANCES
    }
    category_counts: dict[str, int] = {}

    for r in runs:
        if not _image_exists(r):
            continue
        ocr = _get_ocr_cached(r.id, r.source_image)
        stats = ocr.get("stats", {})

        for field, (tol_strict, tol_lenient) in FIELD_TOLERANCES.items():
            db_attr = STAT_TO_DB_ATTR[field]
            gt = getattr(r, db_attr)
            ocr_entry = stats.get(field, {})
            ok, msg, cat = _check_field(field, gt, ocr_entry, tol_strict, tol_lenient)
            if ok:
                passed[field] += 1
            else:
                errors[field].append((r.id, gt, ocr_entry.get("value")))
                if cat:
                    category_counts[cat] = category_counts.get(cat, 0) + 1

    print()
    if len(_OCR_CACHE) < total:
        print(f"Note: {len(_OCR_CACHE)}/{total} runs analyzed in this process")
    print(f"{'Field':<20} {'Accuracy':>10} {'Total':>6}")
    print("-" * 38)
    for field in FIELD_TOLERANCES:
        n = passed[field]
        errs = errors[field]
        pct = f"{n / total * 100:5.1f}%" if total else "N/A"
        print(f"{field:<20} {pct:>10} {total:>6}")
        for rid, gt, ov in errs[:3]:
            print(f"  - id={rid}: expected={gt}, OCR={ov}")
        if len(errs) > 3:
            print(f"  - ... and {len(errs) - 3} more")

    if category_counts:
        print()
        print(f"{'Category':<20} {'Count':>6}")
        print("-" * 28)
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"{cat:<20} {count:>6}")

    all_ok = 0
    for r in runs:
        if not _image_exists(r):
            continue
        ocr = _get_ocr_cached(r.id, r.source_image)
        stats = ocr.get("stats", {})
        ok = True
        for field, (tol_strict, tol_lenient) in FIELD_TOLERANCES.items():
            db_attr = STAT_TO_DB_ATTR[field]
            gt = getattr(r, db_attr)
            ocr_entry = stats.get(field, {})
            passed_field, _, _ = _check_field(
                field, gt, ocr_entry, tol_strict, tol_lenient
            )
            if not passed_field:
                ok = False
                break
        if ok:
            all_ok += 1

    print()
    if total:
        print(
            f"Overall: {all_ok}/{total} runs fully correct "
            f"({all_ok / total * 100:.1f}%)"
        )
