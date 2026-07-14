"""Debug id=26 with latest code."""
import sys
sys.path.insert(0, '.')
from ttr_tracker import ocr
import importlib
importlib.reload(ocr)

result = ocr.extract_stats('test_images/ocr_test_26_50950_51.png')
print(f'pieces: {result["stats"].get("pieces_placed")}')
print(f'score: {result["stats"].get("score")}')

# Manual check of _parse_number
print(f'_parse_number("5 1. 20/8") = {ocr._parse_number("5 1. 20/8")}')
print(f'_parse_number("24, 2418") = {ocr._parse_number("24, 2418")}')
print(f'_parse_number("7, 2411S") = {ocr._parse_number("7, 2411S")}')
