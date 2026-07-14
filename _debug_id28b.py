"""Debug id=28 all_clears with full tracing."""
import sys
sys.path.insert(0, '.')
from ttr_tracker import ocr
import importlib
importlib.reload(ocr)

from ttr_tracker.ocr import extract_stats, preprocess_image, _detect_lines, _find_label_lines, _find_all_clears_value, _focused_number_near_label, _set_result
from PIL import Image
import numpy as np

path = 'test_images/ocr_test_28_113490_87.png'
img = Image.open(path)
proc = preprocess_image(img)
lines = _detect_lines(proc)
labeled = _find_label_lines(lines)

print("=== Step by step ===")
# Simulate extraction for all_clears
result = {"stats": {}}
for key, anchor in labeled:
    if key == 'all_clears':
        print(f'ALL CLEARS anchor: y={anchor["top"]}')
        val = _find_all_clears_value(lines, anchor)
        print(f'  _find_all_clears_value: {val}')
        if val is None:
            val = _focused_number_near_label(img, anchor, preprocessed=proc)
            print(f'  _focused_number_near_label: {val}')
        _set_result(result, "all_clears", val, "high" if val is not None else "missing")
        print(f'  After _set_result: {result["stats"].get("all_clears")}')

print(f'Before cross_validate: {result["stats"].get("all_clears")}')
ocr.cross_validate(result)
print(f'After cross_validate: {result["stats"].get("all_clears")}')

# Now check pieces
for key, anchor in labeled:
    if key == 'pieces_placed':
        from ttr_tracker.ocr import _value_below, _parse_value_and_rate
        val_line = _value_below(lines, anchor)
        print(f'PIECES val_line: text="{val_line["text"]}", y={val_line["top"]}')
        pieces_val, pps_val, _ = _parse_value_and_rate(val_line)
        print(f'pieces={pieces_val}, pps={pps_val}')
