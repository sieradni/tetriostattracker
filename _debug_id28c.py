"""Trace extract_stats for id=28 to find where all_clears becomes 0."""
import sys
sys.path.insert(0, '.')
from ttr_tracker import ocr
import importlib
importlib.reload(ocr)

from ttr_tracker.ocr import preprocess_image, _detect_lines, _find_label_lines, _find_all_clears_value, _focused_number_near_label, _set_result
from PIL import Image
import numpy as np

# Monkey-patch cross_validate to trace
_cv = ocr.cross_validate
def debug_cv(result):
    ac_before = result["stats"].get("all_clears", {}).get("value")
    _cv(result)
    ac_after = result["stats"].get("all_clears", {}).get("value")
    if ac_before != ac_after:
        print(f"  cross_validate changed all_clears: {ac_before} -> {ac_after}")
ocr.cross_validate = debug_cv

# Monkey-patch _set_result to trace
_orig_set = ocr._set_result
def debug_set(result, field, value, conf, warnings=None):
    if field == 'all_clears':
        print(f"  _set_result(all_clears, value={value!r}, conf={conf!r})")
    _orig_set(result, field, value, conf, warnings)
ocr._set_result = debug_set

# Monkey-patch cross_validate to print step by step
print("=== Running extract_stats for id=28 ===")
result = ocr.extract_stats('test_images/ocr_test_28_113490_87.png')
print(f"Final all_clears: {result['stats'].get('all_clears')}")
