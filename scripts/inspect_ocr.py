"""Dump the exact images fed to Tesseract for the all-clears reading, so they
can be inspected manually.

For each given run id, writes to ocr_inspect/<id>/:
  - 00_context.png          original screenshot, cropped around ALL CLEARS
  - 01_preprocessed.png     the binarised image used for the full line pass
  - fallback_cropN_raw.png  the raw crop (before upscale) for _ocr_all_clears_fallback
  - fallback_cropN_tess.png the upscaled inverted image actually sent to Tesseract
  - focused_cropN_raw.png / _tess.png  same for _focused_number_near_label
and prints the raw OCR text Tesseract returns for each tess image.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path when run from scripts/
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ttr_tracker.database import Database
from ttr_tracker import ocr
from PIL import Image
import numpy as np

db = Database("data/tracker.db")
with db.session() as s:
    from ttr_tracker.database import PartialRunRow
    rows = {r.id: r for r in s.query(PartialRunRow).filter(PartialRunRow.source_image.isnot(None)).all()}

ids = [int(a) for a in sys.argv[1:]] or [38, 35, 42, 24]


def tess_inputs_for_crop(crop_img, thresholds=(40, 60)):
    """Return list of (name, PIL) images actually fed to Tesseract for a crop."""
    w, h = crop_img.size
    scale = max(4.0, 600 / w)
    big = crop_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    arr = np.array(big.convert("L"), dtype=np.uint8)
    out = [("inv", Image.fromarray((255 - arr).astype(np.uint8), mode="L"))]
    for thr in thresholds:
        t = np.where((255 - arr) > thr, 255, 0).astype(np.uint8)
        out.append((f"thr{thr}", Image.fromarray(t, mode="L")))
    return out


for rid in ids:
    r = rows[rid]
    img = Image.open(r.source_image)
    proc = ocr.preprocess_image(img)
    lines = ocr._detect_lines(proc)
    ac = None
    for key, a in ocr._find_label_lines(lines):
        if key == "all_clears":
            ac = a
    if ac is None:
        print(f"id={rid}: no ALL CLEARS anchor")
        continue
    l, t, rt, b = ac["left"], ac["top"], ac["right"], ac["bottom"]
    label_w = rt - l
    label_h = b - t
    out_dir = Path("ocr_inspect") / str(rid)
    out_dir.mkdir(parents=True, exist_ok=True)

    # context
    ctx = img.crop((max(0, l - 60), max(0, t - 30), min(img.width, rt + 220), min(img.height, b + 260)))
    ctx.save(out_dir / "00_context.png")
    proc.save(out_dir / "01_preprocessed.png")

    print(f"\n===== id={rid}  GT all_clears={r.all_clears}  anchor x={l}..{rt} y={t}..{b}  img={img.size}")

    # _ocr_all_clears_fallback crops (use the real helper so inspect reflects code)
    proc = ocr.preprocess_image(img)
    scale = proc.width / img.width
    rect = ocr._all_clears_value_rect(ac, img.width, img.height, proc, scale)
    fb_crops = [rect, rect]
    for i, (cl, ct, cr, cb) in enumerate(fb_crops, 1):
        crop = img.crop((cl, ct, cr, cb))
        crop.save(out_dir / f"fallback_crop{i}_raw.png")
        for name, timg in tess_inputs_for_crop(crop):
            timg.save(out_dir / f"fallback_crop{i}_{name}.png")
            d = ocr.pytesseract.image_to_data(timg, output_type=ocr.pytesseract.Output.DICT,
                config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789")
            txt = [d["text"][j] for j in range(len(d["text"])) if d["text"][j].strip()]
            print(f"  fallback#{i} {name}: {txt}")
            # also non-whitelist
            d2 = ocr.pytesseract.image_to_data(timg, output_type=ocr.pytesseract.Output.DICT, config="--psm 7 --oem 3")
            txt2 = [d2["text"][j] for j in range(len(d2["text"])) if d2["text"][j].strip()]
            print(f"  fallback#{i} {name} (no wl): {txt2}")

    # _focused_number_near_label crops (use the real geometry)
    fn_crops = [rect, rect]
    gray = img.convert("L")
    for i, (cl, ct, cr, cb) in enumerate(fn_crops, 1):
        crop = gray.crop((cl, ct, cr, cb))
        crop.save(out_dir / f"focused_crop{i}_raw.png")
        w, h = crop.size
        scale = max(3.0, 600 / w)
        big = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        arr = np.array(big, dtype=np.uint8)
        inv = Image.fromarray((255 - arr).astype(np.uint8), mode="L")
        inv.save(out_dir / f"focused_crop{i}_tess.png")
        d = ocr.pytesseract.image_to_data(inv, output_type=ocr.pytesseract.Output.DICT,
            config="--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789")
        txt = [d["text"][j] for j in range(len(d["text"])) if d["text"][j].strip()]
        print(f"  focused#{i} tess: {txt}")

    print(f"  saved to {out_dir}")


