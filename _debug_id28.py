"""Debug id=28 all_clears issue."""
import sys
sys.path.insert(0, '.')
from ttr_tracker.ocr import extract_stats, preprocess_image, _detect_lines, _find_label_lines, _find_all_clears_value, _focused_number_near_label
from PIL import Image

path = 'test_images/ocr_test_28_113490_87.png'
img = Image.open(path)
proc = preprocess_image(img)
lines = _detect_lines(proc)
labeled = _find_label_lines(lines)

for key, anchor in labeled:
    if key == 'all_clears':
        print(f'ALL CLEARS anchor: y={anchor["top"]}, text="{anchor["text"]}"')
        print(f'  left={anchor["left"]}, right={anchor["right"]}, top={anchor["top"]}, bottom={anchor["bottom"]}')
        v1 = _find_all_clears_value(lines, anchor)
        print(f'  _find_all_clears_value: {v1!r} (type={type(v1).__name__})')
        v2 = _focused_number_near_label(img, anchor, preprocessed=proc)
        print(f'  _focused_number_near_label: {v2!r}')

result = extract_stats(path)
print(f'All_clears in result: {result["stats"].get("all_clears")}')
