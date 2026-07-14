"""Trace all modifications to all_clears."""
import sys, importlib
sys.path.insert(0, '.')
import ttr_tracker.ocr as ocr
importlib.reload(ocr)

# Wrap _set_result to trace all_clears
_orig_set = ocr._set_result
def debug_set(result, field, value, conf, warnings=None):
    if field == 'all_clears':
        import traceback
        print(f"[TRACE] _set_result(all_clears, value={value!r}, conf={conf!r})")
        traceback.print_stack(limit=5)
    _orig_set(result, field, value, conf, warnings)
ocr._set_result = debug_set

result = ocr.extract_stats('test_images/ocr_test_28_113490_87.png')
print(f'\nFinal: {result["stats"].get("all_clears")}')
