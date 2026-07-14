"""Debug id=21 PPS parsing."""
import sys, glob
sys.path.insert(0, '.')
from ttr_tracker.ocr import (
    extract_stats, preprocess_image, _detect_lines, _find_label_lines,
    _value_below, _parse_value_and_rate, _focused_rate_right, RE_PPS,
    _find_all_clears_value, _focused_number_near_label,
)
from PIL import Image

# Load image for id=21
path = 'test_images/ocr_test_21_224_7.png'
img = Image.open(path)
print(f'Image size: {img.size}')
proc = preprocess_image(img)
lines = _detect_lines(proc)
labeled = _find_label_lines(lines)

for key, anchor in labeled:
    if key == 'pieces_placed':
        val_line = _value_below(lines, anchor)
        print(f'PIECES val_line: text="{val_line["text"]}" y={val_line["top"]}')
        pieces_val, pps_val, _ = _parse_value_and_rate(val_line)
        print(f'Initial parse: pieces={pieces_val}, pps={pps_val}')
        
        # What does focused re-OCR return?
        focused_pps = _focused_rate_right(img, val_line, RE_PPS, 10)
        print(f'Focused PPS: {focused_pps}')
        
        # Check the raw text right side
        gray = img.convert("L")
        right_crop = gray.crop((
            max(0, val_line["right"] - 10),
            max(0, val_line["top"] - 15),
            min(img.size[0], val_line["right"] + 200),
            min(img.size[1], val_line["bottom"] + 15),
        ))
        right_crop.save('test_images/_debug_id21_right_crop.png')
        
        # Also scale and save
        w2, h2 = right_crop.size
        scale = max(4.0, 600 / w2)
        big = right_crop.resize((int(w2 * scale), int(h2 * scale)), Image.LANCZOS)
        inv = Image.eval(big, lambda x: 255 - x)
        inv.save('test_images/_debug_id21_right_upscaled.png')

# Also check all_clears
result = extract_stats(path)
print(f'Full result: {result}')
