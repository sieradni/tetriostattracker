"""Trace cross_validate all_clears change."""
import sys, importlib
sys.path.insert(0, '.')
import ttr_tracker.ocr as ocr
importlib.reload(ocr)

result = ocr.extract_stats('test_images/ocr_test_28_113490_87.png')
stats = result['stats']
print(f'spp entry: {stats.get("spp")}')
print(f'pieces entry: {stats.get("pieces_placed")}')
print(f'all_clears entry: {stats.get("all_clears")}')
print(f'time_left entry: {stats.get("time_left")}')

# Manually check what cross_validate does
s = stats
ac = s.get("all_clears", {}).get("value")
print(f'ac = {ac}')
pv = s.get("pieces_placed", {}).get("value")
print(f'pv = {pv}')
sv = s.get("spp", {}).get("value")
print(f'sv = {sv}')
print(f'Check pv < 5: {pv < 5 if pv is not None else "None"}')
print(f'Check sv < 50: {sv < 50 if sv is not None else "None"}')
print(f'Condition: {(pv is not None and pv < 5) or (sv is not None and sv < 50)}')
