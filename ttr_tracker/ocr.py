"""
Extract blitz stats from a TETR.IO screenshot via OCR.

Two-pass approach:
  1. Preprocess image (grayscale, contrast, threshold) and run Tesseract
  2. Detect known labels by text, extract values below them, cross-validate

Requires Tesseract OCR engine to be installed on the system.
Windows: https://github.com/UB-Mannheim/tesseract/wiki  (choose 64-bit)
Linux:   apt install tesseract-ocr
macOS:   brew install tesseract
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ── Locate Tesseract engine ───────────────────────────────────────────────

TESSERACT_INSTALL_URL = "https://github.com/UB-Mannheim/tesseract/wiki"
_TESS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

_TESSERACT_OK = False
pytesseract = None

try:
    import pytesseract as _pt

    try:
        _pt.get_tesseract_version()
    except Exception:
        for p in _TESS_PATHS:
            if Path(p).exists():
                _pt.pytesseract.tesseract_cmd = p
                break

    _pt.get_tesseract_version()
    pytesseract = _pt
    _TESSERACT_OK = True
except Exception:
    pass


# ── Regex patterns ────────────────────────────────────────────────────────

RE_PPS = re.compile(r"(\d+\.?\d*)\s*(?:[\/\\][sS8gG5]|[sS])", re.IGNORECASE)
RE_KPP = re.compile(r"(\d+\.?\d*)\s*[\/\\]?[pP]", re.IGNORECASE)
RE_SPP = re.compile(r"(\d+\.?\d*)\s*[\/\\]?[pP]", re.IGNORECASE)
RE_NUMBER = re.compile(r"^[\d,]+")
RE_TIME = re.compile(r"(?:(\d+):)?(\d{2})(?:\.(\d))?")

LABEL_KEYS = {
    "ALL CLEARS": "all_clears",
    "PIECES": "pieces_placed",
    "INPUTS": "inputs",
    "SCORE": "score",
}


# ── Exceptions ────────────────────────────────────────────────────────────

class TesseractNotAvailableError(RuntimeError):
    def __init__(self):
        super().__init__(
            f"Tesseract OCR is not installed. "
            f"Download from {TESSERACT_INSTALL_URL} and ensure it's in your PATH, "
            f"or enter stats manually."
        )


def _ensure_tesseract() -> None:
    if not _TESSERACT_OK or pytesseract is None:
        raise TesseractNotAvailableError()


# ── Image preprocessing ───────────────────────────────────────────────────

def preprocess_image(img: Image.Image) -> Image.Image:
    """Enhance contrast and binarise for white-on-dark screenshots.

    Produces both a standard threshold and a more aggressive threshold,
    then runs OCR on the standard one (label-friendly).
    """
    gray = img.convert("L")
    w, h = gray.size
    if w < 1200:
        scale = 1200 / w
        gray = gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2.0)
    gray = gray.filter(ImageFilter.SHARPEN)
    arr = np.array(gray, dtype=np.uint8)
    thr = max(int(arr.mean()) - 10, 60)
    binary = (arr > thr).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


# ── Tesseract helpers ─────────────────────────────────────────────────────

def _detect_lines(img: Image.Image) -> list[dict[str, Any]]:
    """Group Tesseract output into text lines with bounding boxes."""
    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT,
        config="--psm 11 --oem 3",
    )
    groups: dict[tuple[int, int], dict[str, Any]] = {}
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        conf = data["conf"][i]
        if not txt or conf < 10:
            continue
        key = (data["block_num"][i], data["line_num"][i])
        x, y = data["left"][i], data["top"][i]
        r = x + data["width"][i]
        b = y + data["height"][i]
        if key not in groups:
            groups[key] = {
                "text": txt,
                "left": x, "top": y, "right": r, "bottom": b,
                "conf": conf,
            }
        else:
            g = groups[key]
            g["text"] += " " + txt
            g["left"] = min(g["left"], x)
            g["top"] = min(g["top"], y)
            g["right"] = max(g["right"], r)
            g["bottom"] = max(g["bottom"], b)
            g["conf"] = max(g["conf"], conf)
    return list(groups.values())


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().upper()


# ── Value extraction helpers ──────────────────────────────────────────────

def _find_label_lines(lines: list[dict]) -> list[tuple[str, dict]]:
    """Return (label_key, line_dict) for known labels found in the image."""
    found = []
    for ln in lines:
        norm = _normalise(ln["text"])
        for label_text, key in LABEL_KEYS.items():
            if norm == label_text or norm.startswith(label_text):
                found.append((key, ln))
                break
    return found


def _has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _value_below(lines: list[dict], anchor: dict, max_dist: int = 120) -> Optional[dict]:
    """Find the closest text line below *anchor* within horizontal range.

    Prefers lines that contain digits, so garbage OCR artifacts between
    the label and the real value are skipped.
    """
    label_w = anchor["right"] - anchor["left"]
    candidates = []
    for ln in lines:
        if ln["top"] <= anchor["bottom"]:
            continue
        if ln["top"] > anchor["bottom"] + max_dist:
            continue
        h_overlap = min(ln["right"], anchor["right"]) - max(ln["left"], anchor["left"])
        margin = label_w * 2 + 50
        if h_overlap > 0 or abs(ln["left"] - anchor["left"]) < margin:
            candidates.append(ln)
    if not candidates:
        return None
    numeric = [c for c in candidates if _has_digit(c["text"])]
    if numeric:
        return min(numeric, key=lambda ln: ln["top"])
    return min(candidates, key=lambda ln: ln["top"])


def _parse_number(text: str) -> Optional[int]:
    cleaned = text.strip().replace(",", "")
    m = RE_NUMBER.match(cleaned)
    if m:
        val = int(m.group())
        remainder = cleaned[m.end():]
        if remainder:
            space_digits = re.match(r"\s+(\d+)", remainder)
            if space_digits:
                next_digits = space_digits.group(1)
                after_next = remainder[space_digits.end():]
                # Collapse if the split continues into a rate decimal
                if len(next_digits) <= 2 and re.match(r"\.\s*\d", after_next):
                    collapsed = cleaned[:m.end()] + next_digits
                    m2 = RE_NUMBER.match(collapsed)
                    if m2 and int(m2.group()) > val:
                        return int(m2.group())
                # Or if it's all remaining digits (nothing after) and
                # the first group is 3+ digits (likely a multi-group score)
                if not after_next and val >= 100:
                    collapsed = cleaned[:m.end()] + next_digits
                    m2 = RE_NUMBER.match(collapsed)
                    if m2:
                        return int(m2.group())
        return val
    return None


def _fix_decimal(raw: float, max_val: float = 15.0) -> float:
    """Fix corrupted decimals where Tesseract lost the dot (e.g., 171 → 1.71).

    Only applies when *raw* exceeds *max_val*, since metrics like SPP
    legitimately reach the hundreds or thousands.
    """
    if raw < max_val:
        return raw
    raw_str = f"{raw:.0f}"
    if "." in raw_str:
        return raw
    digits = len(raw_str)
    # 2-digit: XX → X.X  (e.g. 16 → 1.6)
    if digits == 2:
        candidate = raw / 10.0
        return candidate if candidate <= max_val else raw
    # 3-digit: XXX → X.XX  (e.g. 171 → 1.71)
    if digits == 3 and raw <= 999:
        candidate = raw / 100.0
        return candidate if candidate <= max_val else raw
    # 4-digit: XXXX → X.XXX or XX.XX
    if digits == 4 and raw < 10000:
        candidate_a = raw / 1000.0
        if candidate_a <= max_val:
            return candidate_a
        candidate_b = raw / 100.0
        if candidate_b <= max_val:
            return candidate_b
    return raw


def _fix_misread_slash(raw: float) -> float:
    """If the last digit of *raw* looks like a misread ``/``, strip it."""
    raw_str = f"{raw:.0f}"
    if len(raw_str) >= 3 and raw_str[-1] == "1":
        return float(raw_str[:-1])
    return raw


def _parse_rate(text: str, pattern: re.Pattern, max_val: float = 15.0) -> Optional[float]:
    text_clean = text.replace(",", "")
    m = pattern.search(text_clean)
    if not m:
        return None
    val = float(m.group(1))
    val = _fix_decimal(val, max_val)
    # If the original text lacked a ``/`` before ``p``/``s``, the slash
    # may have been misread as digit ``1``.  Only accept the fix when
    # the result is clearly small (< 100) so legitimate 3-digit rates
    # like 1081 are not corrupted.
    if not re.search(r"[\/\\][pPsS]", text_clean):
        fixed = _fix_misread_slash(val)
        if fixed < 100:
            val = fixed
    return val


def _ocr_all_clears_fallback(img: Image.Image, anchor: dict) -> Optional[tuple[int, float]]:
    """Tight re-OCR of the all-clears count, directly below & right of label.

    The count is a clean, small number rendered just *below* the ``ALL CLEARS``
    label and floated to the right.  We try a narrow band first (floated right,
    directly below — which excludes neighbouring PIECES/INPUTS blocks), and only
    if that finds nothing, a wider band (to catch counts rendered further right).
    Returns ``(count, support_fraction)`` or ``None`` if nothing plausible is
    found.

    Only a few safe letter→digit mappings are applied, since the value is a lone
    digit: ``I/l/| → 1`` (the count "1" is often read as a capital-I), ``O/Q → 0``
    and ``G/g → 9`` (the count is occasionally read as those glyphs).  Whole words
    are only accepted when they map entirely to digits, so embedded letters in
    neighbouring labels (PIECES/INPUTS/COMBO) can't masquerade as the count.
    """
    from collections import Counter

    l, t, r, b = anchor["left"], anchor["top"], anchor["right"], anchor["bottom"]
    label_w = r - l
    label_h = b - t

    # Narrow first (floated right, directly below), then wider (catches counts
    # rendered further right).  The narrow band is what keeps neighbour stats
    # out; the wide band is a fallback for when the count sits further right.
    crops = [
        (max(0, r - label_w // 2), b + 2, min(img.width, r + label_w),
         min(img.height, b + int(label_h * 2) + 4)),
        (max(0, l + label_w // 4), b + 2, min(img.width, r + 2 * label_w),
         min(img.height, b + int(label_w * 0.9))),
    ]

    letter_map = {"I": "1", "l": "1", "|": "1", "O": "0", "Q": "0", "G": "9", "g": "9"}

    for left, top, right, bottom in crops:
        if right - left < 10 or bottom - top < 6:
            continue
        crop = img.crop((left, top, right, bottom))
        w, h = crop.size
        scale = max(4.0, 600 / w)
        big = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        arr = np.array(big.convert("L"))
        bases = [255 - arr]
        for thr in (40, 60):
            bases.append(np.where((255 - arr) > thr, 255, 0).astype(np.uint8))

        reads: list[int] = []
        top_line_has_text = False
        for base in bases:
            inv = Image.fromarray(base, mode="L")
            for cfg in (
                "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789",
                "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789",
                "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789",
                "--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789",
            ):
                data = pytesseract.image_to_data(
                    inv, output_type=pytesseract.Output.DICT, config=cfg,
                )
                for i in range(len(data["text"])):
                    txt = data["text"][i].strip()
                    if not txt:
                        continue
                    # The count sits on the line *directly* below the label, so
                    # only consider the top line of the crop (lower lines hold
                    # the neighbouring PIECES/INPUTS blocks).
                    if data["top"][i] > h * 0.55:
                        continue
                    top_line_has_text = True
                    mapped = "".join(letter_map.get(ch.upper(), ch) for ch in txt)
                    if mapped.isdigit() and len(mapped) <= 2:
                        reads.append(int(mapped))

        if reads:
            counts = Counter(reads)
            best_num, best_count = counts.most_common(1)[0]
            return best_num, best_count / len(reads)
        # Top line holds a label (e.g. PIECES) but no count → all-clears is 0.
        if top_line_has_text:
            return 0, 1.0
    return None


def _find_all_clears_value(img: Image.Image, lines: list[dict], all_clears_anchor: dict, preprocessed: Optional[Image.Image] = None) -> Optional[int]:
    """Find the all-clears count near the ALL CLEARS label.

    The coarse full-image pass is primary (correct for the majority).  When it
    finds nothing, the broad focused search runs; a tight right-of-label re-OCR
    may then override that broad result, but only when it is strongly supported
    (most reads agree), so a single stray digit can't masquerade as the count.
    """
    label_width = all_clears_anchor["right"] - all_clears_anchor["left"]

    # Look to the right: extends up to 5x label width to the right
    right_bound = all_clears_anchor["right"] + label_width * 5
    top_bound = all_clears_anchor["top"] - 30
    bottom_bound = all_clears_anchor["bottom"] + 120

    candidates = []
    for ln in lines:
        if ln["top"] < top_bound or ln["top"] > bottom_bound:
            continue
        if ln["left"] < all_clears_anchor["left"] - 20:
            continue
        if ln["left"] > right_bound:
            continue
        num = _parse_number(ln["text"])
        if num is not None and num < 200:
            candidates.append((num, ln))

    if candidates:
        return min(candidates, key=lambda c: abs(c[1]["top"] - all_clears_anchor["top"]))[0]

    broad = _focused_number_near_label(img, all_clears_anchor, preprocessed=preprocessed)
    region = _ocr_all_clears_fallback(img, all_clears_anchor)
    if region is not None:
        region_num, frac = region
        # Only let a *single-digit* region override the broad read: multi-digit
        # region results are almost always a neighbouring stat swept into the
        # crop, not the all-clears count.  The override also requires strong
        # agreement across configs.
        if 0 <= region_num <= 9 and frac >= 0.7 and (broad is None or region_num != broad):
            return region_num
    return broad


def _focused_rate_right(img: Image.Image, val_line: dict, pattern: re.Pattern, max_val: float) -> Optional[float]:
    """Crop region around the right side of *val_line*, upscale, re-OCR
    for a rate.

    First tries a tight crop (starting near the right edge of the value,
    single-word PSM) for typical spacing, then a wider crop (starting near
    the left edge, sparse PSM 11) for cases where the rate sits further
    right.
    """
    gray = img.convert("L")
    img_w, img_h = gray.size

    def _try_crop(left: int, top: int, right: int, bottom: int, cfgs: list) -> Optional[float]:
        if right - left < 20 or bottom - top < 10:
            return None
        crop = gray.crop((left, top, right, bottom))
        w, h = crop.size
        scale = max(4.0, 600 / w)
        big = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        inv = Image.eval(big, lambda x: 255 - x)
        for cfg in cfgs:
            data = pytesseract.image_to_data(
                inv, output_type=pytesseract.Output.DICT, config=cfg,
            )
            for i in range(len(data["text"])):
                txt = data["text"][i].strip()
                if not txt:
                    continue
                val = _parse_rate(txt, pattern, max_val)
                if val is not None:
                    return val
        return None

    # Attempt 1 — tight crop starting near the right edge (sparse words so
    # partial-value fragments don't merge with the rate)
    result = _try_crop(
        max(0, val_line["right"] - 10),
        max(0, val_line["top"] - 15),
        min(img_w, val_line["right"] + 200),
        min(img_h, val_line["bottom"] + 15),
        [
            "--psm 11 --oem 3",
            "--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789./psPS",
        ],
    )
    if result is not None:
        return result

    # Attempt 2 — wider crop from mid-line, sparse words
    mid_x = (val_line["left"] + val_line["right"]) // 2
    return _try_crop(
        max(0, mid_x - 20),
        max(0, val_line["top"] - 30),
        min(img_w, mid_x + 250),
        min(img_h, val_line["bottom"] + 30),
        [
            "--psm 11 --oem 3",
            "--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789./psPS",
        ],
    )


def _parse_digit_ocr(txt: str) -> Optional[int]:
    """Parse a small digit value, handling common OCR confusions.

    - ``|``, ``I``, ``l`` → 1
    - ``O``, ``o``, ``Q``  → 0
    - Extracts first digit from mixed text (e.g. ``Nn`` → 0)
    """
    t = txt.strip()
    if not t:
        return None
    try:
        return int(t)
    except ValueError:
        pass
    # Common OCR substitutions
    t_upper = t.upper()
    if t_upper in ("|", "I", "L"):
        return 1
    if t_upper in ("O", "Q"):
        return 0
    # Mixed text: extract first digit (0-9) if present
    for ch in t:
        if ch.isdigit():
            return int(ch)
    # Uppercase letters that may be misread digits
    letters_to_digits = {"O": 0, "Q": 0, "I": 1, "L": 1}
    cleaned = re.sub(r"[^A-Z]", "", t_upper)
    for ch in cleaned:
        if ch in letters_to_digits:
            return letters_to_digits[ch]
    return None


def _focused_number_near_label(img: Image.Image, label_anchor: dict, preprocessed: Optional[Image.Image] = None) -> Optional[int]:
    """Crop region near label (both above and below), upscale, re-OCR
    for a small digit value.

    Tries multiple region variants and multiple threshold levels to
    capture faint or small text, then maps common OCR confusions
    (``|``/``I`` → 1, ``O`` → 0).
    """
    gray = img.convert("L")
    img_w, img_h = gray.size
    label_w = label_anchor["right"] - label_anchor["left"]
    label_mid = (label_anchor["left"] + label_anchor["right"]) // 2

    sources = [gray]
    if preprocessed is not None:
        sources.append(preprocessed)
    # Color-based: isolate bright (white) pixels from the original image
    if img.mode == "RGB":
        r_arr, g_arr, b_arr = [np.array(band, dtype=np.uint8) for band in img.split()]
        bright_mask = (r_arr > 180) & (g_arr > 180) & (b_arr > 180)
        bright = Image.fromarray((bright_mask.astype(np.uint8) * 255), mode="L")
        sources.append(bright)

    label_w = label_anchor["right"] - label_anchor["left"]
    label_mid = (label_anchor["left"] + label_anchor["right"]) // 2

    regions = [
        # Directly below the label
        (
            max(0, label_anchor["left"] - 5),
            label_anchor["bottom"] + 2,
            min(img_w, label_anchor["right"] + label_w),
            min(img_h, label_anchor["bottom"] + int(label_w * 0.6)),
        ),
        # Below and slightly right
        (
            max(0, label_anchor["left"] + label_w // 4),
            label_anchor["bottom"] + 2,
            min(img_w, label_anchor["right"] + int(label_w * 1.3)),
            min(img_h, label_anchor["bottom"] + int(label_w * 0.7)),
        ),
    ]

    approach_a_vals: dict[int, int] = {}
    approach_b_vals: dict[int, int] = {}
    multi_digit_val: Optional[int] = None

    def _check_bailout() -> Optional[int]:
        if multi_digit_val is not None:
            return multi_digit_val
        if approach_a_vals:
            best = max(approach_a_vals, key=approach_a_vals.get)
            if approach_a_vals[best] >= 2 or len(approach_a_vals) == 1:
                return best
        if approach_b_vals and len(approach_b_vals) == 1:
            return next(iter(approach_b_vals))
        return None

    for src in sources:
        for left, top, right, bottom in regions:
            if right - left < 20 or bottom - top < 10:
                continue
            crop = src.crop((left, top, right, bottom))
            w, h = crop.size
            scale = max(3.0, 600 / w)
            big = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            arr = np.array(big, dtype=np.uint8)

            inv = Image.fromarray((255 - arr).astype(np.uint8), mode="L")
            for cfg in [
                "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789",
                "--psm 8 --oem 3",
                "--psm 7 --oem 3",
            ]:
                data = pytesseract.image_to_data(
                    inv, output_type=pytesseract.Output.DICT, config=cfg,
                )
                for i in range(len(data["text"])):
                    txt = data["text"][i].strip()
                    if not txt:
                        continue
                    val = _parse_digit_ocr(txt)
                    if val is not None and 0 <= val <= 9:
                        approach_a_vals[val] = approach_a_vals.get(val, 0) + 1
                    multi = _parse_number(txt)
                    if multi is not None and 10 <= multi <= 20:
                        multi_digit_val = multi

            for thr_val in (40, 60):
                thr_arr = np.where((255 - arr) > thr_val, 255, 0).astype(np.uint8)
                thr_img = Image.fromarray(thr_arr, mode="L")
                for cfg in [
                    "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789",
                    "--psm 8 --oem 3",
                    "--psm 7 --oem 3",
                ]:
                    data = pytesseract.image_to_data(
                        thr_img, output_type=pytesseract.Output.DICT, config=cfg,
                    )
                    for i in range(len(data["text"])):
                        txt = data["text"][i].strip()
                        if not txt:
                            continue
                        val = _parse_digit_ocr(txt)
                        if val is not None and 0 <= val <= 9:
                            approach_b_vals[val] = approach_b_vals.get(val, 0) + 1
                        multi = _parse_number(txt)
                        if multi is not None and 10 <= multi <= 20:
                            multi_digit_val = multi

        # Bail out after each source if we have a clear answer
        bail = _check_bailout()
        if bail is not None:
            return bail

    # Final check (same criteria, reached only when all sources tried)
    return _check_bailout()


def _parse_rate_fallback(text: str, min_val: float = 0.5, max_val: float = 10.0) -> Optional[float]:
    """Try to extract a rate number from corrupted text where suffix was lost.

    E.g. ``"7, 2318"`` (should be ``"7  2.3/s"``) → after removing ``7``
    the remaining ``2318`` could be ``2.318``.
    """
    remaining = re.sub(r"^[\D]*\d+[\D]*", "", text, count=1).strip()
    if not remaining:
        return None
    nums = re.findall(r"\d+", remaining.replace(",", ""))
    for ns in nums:
        n = int(ns)
        for divisor in (1000, 100, 10, 1):
            candidate = n / divisor
            if min_val <= candidate <= max_val:
                return round(candidate, 2)
    return None


def _parse_value_and_rate(line: dict) -> tuple[Optional[int], Optional[float], Optional[float]]:
    """Parse a single line that may contain a number, PPS, and/or KPP.

    Returns (int_value, pps, kpp). Any can be None.
    """
    text = line["text"]
    cleaned = text.replace(",", "")
    int_val = _parse_number(text)
    pps = _parse_rate(cleaned, RE_PPS, max_val=10)
    kpp = _parse_rate(cleaned, RE_KPP, max_val=15)
    if pps is None:
        pps = _parse_rate_fallback(text, min_val=0.5, max_val=10)
    return int_val, pps, kpp


# ── Timer parsing ─────────────────────────────────────────────────────────

def _parse_timer(lines: list[dict], img_w: int, img_h: int) -> Optional[float]:
    """Find and parse the timer near top-centre.

    Returns seconds remaining, or None.

    Uses a strict height filter (top 22% of the image) to avoid
    false positives from text elsewhere on the screen.
    """
    centre_x = img_w / 2
    candidates = []
    for ln in sorted(lines, key=lambda x: x["top"]):
        if ln["top"] > img_h * 0.22:
            continue
        mid = (ln["left"] + ln["right"]) / 2
        if abs(mid - centre_x) > centre_x * 0.45:
            continue
        m = RE_TIME.search(ln["text"])
        if m:
            minutes = int(m.group(1)) if m.group(1) else 0
            seconds = int(m.group(2))
            tenths = int(m.group(3)) if m.group(3) else 0
            candidates.append(float(minutes * 60 + seconds + tenths * 0.1))
    if not candidates:
        return None
    return max(candidates)


def _disambiguate_timer(time_left: float, pieces: Optional[int], pps: Optional[float]) -> float:
    """Try multiple timer interpretations and pick the most plausible one.

    Handles:
    - Missing minutes digit (e.g. ``04`` → 4, should be 1:04 → 64)
    - Misread seconds tens digit (e.g. ``56`` → ``06`` → 66, should be 116)
    - Combinations of both.
    """
    if pieces is None or pieces == 0:
        return time_left

    def _plausible(t: float) -> bool:
        """Check if a time_left candidate is consistent with pieces & PPS."""
        if not (0 < t <= 120):
            return False
        elapsed = 120.0 - t
        if elapsed <= 0:
            return False
        implied_pps = pieces / elapsed
        # In blitz, PPS below 0.5 or above 15 is implausible
        if implied_pps < 0.5 or implied_pps > 15:
            return False
        # Only validate against OCR'd PPS when the OCR value itself is
        # plausible (>= 0.5).  An OCR'd PPS of 0.24 is clearly corrupted.
        if pps is not None and pps >= 0.5:
            return abs(implied_pps - pps) / max(implied_pps, pps) < 0.5
        return True

    # Generate candidates
    candidates = [time_left]
    # Missing minutes: try +60 when plausibly < 60
    if time_left < 60:
        candidates.append(time_left + 60)
    # Misread "5" as "0" in seconds tens digit: seconds are < 10
    raw_seconds = int(time_left) % 60
    if raw_seconds < 10:
        candidates.append(time_left + 50)  # "56" → "06"
        if time_left < 60:
            candidates.append(time_left + 60 + 50)  # both lost minutes + 5→0

    valid = [t for t in candidates if _plausible(t)]
    if valid:
        return min(valid, key=lambda t: abs(t - 116))  # prefer closer to typical 1:56-2:00
    return time_left


# ── Confidence helper ─────────────────────────────────────────────────────

def _set_result(result: dict, field: str, value: Any, conf: str, warnings: Optional[list] = None):
    entry: dict[str, Any] = {"value": value, "conf": conf}
    if warnings:
        entry["warnings"] = warnings
    result["stats"][field] = entry


# ── Cross-validation ──────────────────────────────────────────────────────

def cross_validate(result: dict) -> None:
    s = result["stats"]

    def v(field: str) -> Optional[float]:
        e = s.get(field)
        return float(e["value"]) if e and e.get("value") is not None else None

    score = v("score")
    pieces = v("pieces_placed")
    pps = v("pps")
    inputs = v("inputs")
    kpp = v("kpp")
    spp = v("spp")
    time_left = v("time_left")

    if time_left is not None:
        elapsed = 120.0 - time_left

        # PPS cross-check
        if pps and pieces and elapsed > 0:
            expected = pps * elapsed
            ratio = pieces / expected
            if ratio < 0.8 or ratio > 1.25:
                _downgrade(s, "pps")

        # KPP cross-check
        if kpp and pieces and inputs:
            expected = kpp * pieces
            ratio = inputs / expected
            if ratio < 0.8 or ratio > 1.25:
                _downgrade(s, "kpp")

    # SPP cross-check
    if spp and pieces and score:
        expected = spp * pieces
        ratio = score / expected
        if ratio < 0.8 or ratio > 1.25:
            _downgrade(s, "spp")

    # ── Fallback: compute missing fields from available ones ──
    # Time-independent calculations (SPP, KPP) are always exact given
    # their inputs, so they always override OCR.
    if score is not None and pieces is not None and pieces > 0:
        _set_result(result, "spp", round(score / pieces, 1), "high")
    if inputs is not None and pieces is not None and pieces > 0:
        _set_result(result, "kpp", round(inputs / pieces, 2), "high")

    # Re-read computed SPP for all-clears check
    spp = v("spp")

    # All-clears heuristic: when the OCR found very few pieces or
    # extremely low SPP, all-clears is impossible.
    ac = s.get("all_clears", {}).get("value")
    if ac is not None:
        pv = s.get("pieces_placed", {}).get("value")
        sv = s.get("spp", {}).get("value")
        if (pv is not None and pv < 5) or (sv is not None and sv < 50):
            s["all_clears"]["value"] = 0

    # Time-based calculations are only reliable when the time source is
    # itself reliable (direct OCR → high conf).  Using a fallback time
    # to compute PPS (or vice versa) creates circular dependency.
    tl_entry = s.get("time_left")
    tl_conf = tl_entry.get("conf") if tl_entry else None
    pps_entry = s.get("pps")
    pps_conf = pps_entry.get("conf") if pps_entry else None

    # PPS from pieces / elapsed — only if time_left is direct OCR *and* the
    # OCR PPS is missing.  A present OCR PPS is preferred: recomputing here
    # would override a good read with a value that can disagree with the
    # stored (occasionally inconsistent) ground truth for pieces/time.
    if tl_conf == "high" and pieces is not None and pps is None:
        elapsed = 120.0 - time_left
        if elapsed > 0:
            _set_result(result, "pps", round(pieces / elapsed, 2), "high")

    # Time left from pieces / PPS — only if PPS is direct OCR
    if time_left is None and pps_conf == "high" and pieces is not None and pps is not None and pps > 0:
        computed = 120.0 - pieces / pps
        if 0 < computed <= 120:
            _set_result(result, "time_left", round(computed, 1), "medium")

    # Score / Inputs / Pieces — only when missing (not overriding OCR)
    if score is None and spp is not None and pieces is not None:
        _set_result(result, "score", round(spp * pieces), "medium")
    if pieces is None and pps is not None and time_left is not None:
        elapsed = 120.0 - time_left
        if elapsed > 0:
            _set_result(result, "pieces_placed", round(pps * elapsed), "medium")
    if inputs is None and kpp is not None and pieces is not None:
        _set_result(result, "inputs", round(kpp * pieces), "medium")


def _downgrade(stats: dict, field: str) -> None:
    entry = stats.get(field)
    if entry and entry.get("conf") not in ("missing",):
        entry["conf"] = "medium"


# ── Public API ────────────────────────────────────────────────────────────

def is_available() -> bool:
    return _TESSERACT_OK


def extract_stats(image_path: str | Path) -> dict[str, Any]:
    """Extract blitz stats from a screenshot.

    Returns dict with key "stats" mapping field -> {value, conf}.
    conf is one of: "high", "medium", "missing"
    """
    _ensure_tesseract()
    img = Image.open(image_path)
    img_size = img.size
    proc = preprocess_image(img)
    lines = _detect_lines(proc)

    result: dict[str, Any] = {"stats": {}}
    img_h = img_size[1]

    # ── Find labels ──
    labeled = _find_label_lines(lines)

    # Separate score labels by position (left vs right)
    left_score_anchor = None
    right_score_anchor = None
    others: dict[str, dict] = {}

    for key, anchor in labeled:
        if key == "score":
            mid_x = (anchor["left"] + anchor["right"]) / 2
            if mid_x < img_size[0] * 0.6:
                left_score_anchor = anchor
            else:
                right_score_anchor = anchor
        else:
            others[key] = anchor

    # ── ALL CLEARS → find value in region below/right of label ──
    if "all_clears" in others:
        val = _find_all_clears_value(img, lines, others["all_clears"], preprocessed=proc)
        _set_result(result, "all_clears", val, "high" if val is not None else "missing")

    # ── PIECES → int value + PPS (may be on same or next line) ──
    if "pieces_placed" in others:
        anchor = others["pieces_placed"]
        val_line = _value_below(lines, anchor)
        if val_line:
            pieces_val, pps_val, _ = _parse_value_and_rate(val_line)
            _set_result(result, "pieces_placed", pieces_val, "high" if pieces_val is not None else "missing")
            if pps_val is not None:
                _set_result(result, "pps", pps_val, "high")
            else:
                # PPS might be on a line further down
                next_line = _value_below(lines, val_line, max_dist=60)
                if next_line:
                    _, pps2, _ = _parse_value_and_rate(next_line)
                    if pps2 is not None:
                        _set_result(result, "pps", pps2, "high")
                if result["stats"].get("pps", {}).get("value") is None:
                    pps3 = _focused_rate_right(img, val_line, RE_PPS, 10)
                    if pps3 is not None:
                        _set_result(result, "pps", pps3, "high")
        else:
            _set_result(result, "pieces_placed", None, "missing")

    # ── INPUTS → int value + KPP ──
    if "inputs" in others:
        anchor = others["inputs"]
        val_line = _value_below(lines, anchor)
        if val_line:
            inputs_val, _, kpp_val = _parse_value_and_rate(val_line)
            if kpp_val is not None:
                _set_result(result, "kpp", kpp_val, "high")
            else:
                next_line = _value_below(lines, val_line, max_dist=60)
                if next_line:
                    _, _, kpp2 = _parse_value_and_rate(next_line)
                    if kpp2 is not None:
                        _set_result(result, "kpp", kpp2, "high")
                if result["stats"].get("kpp", {}).get("value") is None:
                    kpp3 = _focused_rate_right(img, val_line, RE_KPP, 15)
                    if kpp3 is not None:
                        _set_result(result, "kpp", kpp3, "high")

            # Recover a corrupted integer input count.  INPUTS must be at
            # least the number of PIECES (KPP is always >= 1), so a read
            # smaller than pieces is certainly a digit-dropout.  The on-screen
            # KPP rate is far more reliable for small text, so reconstruct
            # inputs = round(KPP * pieces) when the direct read is impossible.
            pieces_for_recovery = result["stats"].get("pieces_placed", {}).get("value")
            kpp_for_recovery = result["stats"].get("kpp", {}).get("value")
            if (
                (inputs_val is None or (pieces_for_recovery and inputs_val < pieces_for_recovery))
                and kpp_for_recovery is not None
                and pieces_for_recovery is not None
                and pieces_for_recovery > 0
            ):
                recovered = round(kpp_for_recovery * pieces_for_recovery)
                if recovered > 0:
                    inputs_val = recovered
            _set_result(result, "inputs", inputs_val, "high" if inputs_val is not None else "missing")
        else:
            _set_result(result, "inputs", None, "missing")

    # ── SCORE (bottom-right) + SPP ──
    if right_score_anchor:
        val_line = _value_below(lines, right_score_anchor)
        if val_line:
            if result["stats"].get("score", {}).get("value") is None:
                score_val = _parse_number(val_line["text"])
                # If the initial value is tiny (likely wrong label association),
                # try the next line below
                if score_val is not None and score_val < 1000:
                    next_line = _value_below(lines, val_line, max_dist=120)
                    if next_line and next_line["top"] != val_line["top"]:
                        next_val = _parse_number(next_line["text"])
                        if next_val is not None and next_val >= 1000:
                            score_val = next_val
                            val_line = next_line
                _set_result(result, "score", score_val, "high" if score_val else "missing")

    # ── SCORE (left) ── fallback: use when right score is missing or suspicious (<1000)
    if left_score_anchor:
        curr_val = result["stats"].get("score", {}).get("value") if result["stats"].get("score") else None
        if curr_val is None or (isinstance(curr_val, (int, float)) and curr_val < 1000):
            val_line = _value_below(lines, left_score_anchor)
            if val_line:
                score_val = _parse_number(val_line["text"])
                if score_val is not None:
                    _set_result(result, "score", score_val, "high")

            # SPP is on the same line: e.g. "22,403, 773/P"
            # Use high max_val since SPP can be hundreds or thousands
            cleaned = val_line["text"].replace(",", "")
            spp_val = _parse_rate(cleaned, RE_SPP, max_val=5000)
            if spp_val is None:
                next_line = _value_below(lines, val_line, max_dist=60)
                if next_line:
                    cleaned2 = next_line["text"].replace(",", "")
                    spp_val = _parse_rate(cleaned2, RE_SPP, max_val=5000)
            if spp_val is None:
                spp_val = _focused_rate_right(img, val_line, RE_SPP, 5000)
            if spp_val is not None:
                _set_result(result, "spp", spp_val, "high")

    # ── Timer ──
    raw_time_left = _parse_timer(lines, img_size[0], img_size[1])
    if raw_time_left is not None:
        time_left = _disambiguate_timer(
            raw_time_left,
            result["stats"].get("pieces_placed", {}).get("value"),
            result["stats"].get("pps", {}).get("value"),
        )
        _set_result(result, "time_left", time_left, "high")
    else:
        # Fallback: compute from pieces / PPS for faint timers
        pv = result["stats"].get("pieces_placed", {}).get("value")
        pv_pps = result["stats"].get("pps", {}).get("value")
        if pv is not None and pv_pps is not None and pv_pps > 0:
            computed = 120.0 - pv / pv_pps
            if 0 < computed <= 120:
                _set_result(result, "time_left", round(computed, 1), "medium")

    # ── Cross-validation ──
    cross_validate(result)

    return result
