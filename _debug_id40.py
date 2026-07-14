"""Debug id=40 pieces parsing."""
import sys
sys.path.insert(0, '.')
from ttr_tracker import ocr
import importlib
importlib.reload(ocr)

from ttr_tracker.ocr import _parse_number, preprocess_image, _detect_lines, _find_label_lines, _value_below
from PIL import Image

path = 'test_images/ocr_test_40_*.png'
import glob
files = glob.glob(path)
if files:
    path = files[0]

img = Image.open(path)
proc = preprocess_image(img)
lines = _detect_lines(proc)
labeled = _find_label_lines(lines)

for key, anchor in labeled:
    if key == 'pieces_placed':
        print(f'PIECES anchor y={anchor["top"]}')
        val_line = _value_below(lines, anchor)
        if val_line:
            text = val_line["text"]
            print(f'  val_line text="{text}" y={val_line["top"]}')
            print(f'  _parse_number("{text}") = {_parse_number(text)}')
            # Manually trace
            cleaned = text.strip().replace(",", "")
            print(f'  cleaned="{cleaned}"')
            import re
            m = re.match(r"^[\d,]+", cleaned)
            if m:
                print(f'  RE_NUMBER match: "{m.group()}"')
                remainder = cleaned[m.end():]
                print(f'  remainder="{remainder}"')
