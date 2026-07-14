"""Debug id=26 pieces parsing."""
import sys
sys.path.insert(0, '.')
from ttr_tracker import ocr
import importlib
importlib.reload(ocr)

# Check _parse_number for the pieces line
from ttr_tracker.ocr import _parse_number, _value_below, _parse_value_and_rate, preprocess_image, _detect_lines, _find_label_lines
from PIL import Image

path = 'test_images/ocr_test_26_50950_51.png'
img = Image.open(path)
proc = preprocess_image(img)
lines = _detect_lines(proc)
labeled = _find_label_lines(lines)

for key, anchor in labeled:
    if key == 'pieces_placed':
        print(f'PIECES anchor y={anchor["top"]}')
        val_line = _value_below(lines, anchor)
        if val_line:
            print(f'  val_line text="{val_line["text"]}" y={val_line["top"]}')
            # Manually parse
            text = val_line["text"]
            print(f'  _parse_number("{text}") = {_parse_number(text)}')
            pieces, pps, _ = _parse_value_and_rate(val_line)
            print(f'  _parse_value_and_rate -> pieces={pieces}, pps={pps}')

# Also check full result
result = ocr.extract_stats(path)
print(f'Full: pieces={result["stats"].get("pieces_placed")}')
print(f'Full: score={result["stats"].get("score")}')
