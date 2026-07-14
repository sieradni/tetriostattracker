"""OCR accuracy tests: compare extract_stats() output against verified ground truth."""

import os
from pathlib import Path

import pytest

from ttr_tracker.database import Database, PartialRunRow
from ttr_tracker.ocr import extract_stats

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.environ.get("TTR_DB_PATH", str(BASE_DIR / "data" / "tracker.db"))
db = Database(DB_PATH)

# Fields to compare, with comparison tolerance
FIELD_TOLERANCES: dict[str, float] = {
    "score": 0,
    "pieces_placed": 0,
    "pps": 0.05,
    "inputs": 0,
    "kpp": 0.05,
    "spp": 1.0,
    "all_clears": 0,
    "time_left": 1.0,
}

# Map OCR stat key → PartialRunRow attribute name
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


def _get_verified_runs() -> list[PartialRunRow]:
    with db.session() as sess:
        return (
            sess.query(PartialRunRow)
            .filter(PartialRunRow.source_image.isnot(None))
            .order_by(PartialRunRow.id)
            .all()
        )


def _image_exists(run: PartialRunRow) -> bool:
    return run.source_image is not None and Path(run.source_image).is_file()


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def verified_runs():
    return _get_verified_runs()


# ── Test cases ────────────────────────────────────────────────────────────


def pytest_generate_tests(metafunc):
    if "run_and_ocr" in metafunc.fixturenames:
        runs = _get_verified_runs()
        pairs = []
        ids = []
        for r in runs:
            if _image_exists(r):
                pairs.append((r, extract_stats(r.source_image)))
                ids.append(f"id={r.id}")
        metafunc.parametrize("run_and_ocr", pairs, ids=ids)


@pytest.mark.ocr
def test_ocr_fields(run_and_ocr):
    run, ocr_result = run_and_ocr
    stats = ocr_result.get("stats", {})
    failures = []

    for stat_field, db_attr in STAT_TO_DB_ATTR.items():
        tol = FIELD_TOLERANCES[stat_field]
        ground_truth = getattr(run, db_attr)
        ocr_entry = stats.get(stat_field, {})
        ocr_val = ocr_entry.get("value")

        if ocr_val is None and ground_truth is None:
            continue
        if ocr_val is None:
            failures.append(f"{stat_field}: OCR=None, expected={ground_truth}")
            continue
        if ground_truth is None:
            failures.append(f"{stat_field}: OCR={ocr_val}, expected=None")
            continue

        diff = abs(float(ocr_val) - float(ground_truth))
        if diff > tol:
            failures.append(
                f"{stat_field}: OCR={ocr_val}, expected={ground_truth}, diff={diff:.2f} (tolerance={tol})"
            )

    if failures:
        details = "; ".join(failures)
        pytest.fail(f"[id={run.id} score={run.score}] {details}")


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


# ── Summary ───────────────────────────────────────────────────────────────


@pytest.mark.ocr
def test_accuracy_summary():
    """Print a per-field accuracy table."""
    runs = _get_verified_runs()
    total = len(runs)
    if total == 0:
        pytest.skip("No verified runs with source images found")

    passed: dict[str, int] = {f: 0 for f in FIELD_TOLERANCES}
    errors: dict[str, list[tuple[int, float | int, float | int]]] = {
        f: [] for f in FIELD_TOLERANCES
    }

    for r in runs:
        if not _image_exists(r):
            continue
        ocr = extract_stats(r.source_image)
        stats = ocr.get("stats", {})

        for field, tol in FIELD_TOLERANCES.items():
            db_attr = STAT_TO_DB_ATTR[field]
            gt = getattr(r, db_attr)
            ov = stats.get(field, {}).get("value")
            if ov is None or gt is None:
                continue
            if abs(float(ov) - float(gt)) <= tol:
                passed[field] += 1
            else:
                errors[field].append((r.id, gt, ov))

    print()
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

    # Collect overall pass/fail per run
    all_ok = 0
    for r in runs:
        if not _image_exists(r):
            continue
        ocr = extract_stats(r.source_image)
        stats = ocr.get("stats", {})
        ok = True
        for field, tol in FIELD_TOLERANCES.items():
            db_attr = STAT_TO_DB_ATTR[field]
            gt = getattr(r, db_attr)
            ov = stats.get(field, {}).get("value")
            if ov is None or gt is None or abs(float(ov) - float(gt)) > tol:
                ok = False
                break
        if ok:
            all_ok += 1

    print()
    if total:
        print(
            f"Overall: {all_ok}/{total} runs fully correct ({all_ok / total * 100:.1f}%)"
        )
