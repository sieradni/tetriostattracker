"""Debug script to analyze OCR failures."""
import sys, os, json, glob
sys.path.insert(0, '.')
from ttr_tracker.ocr import extract_stats, preprocess_image, _detect_lines, _find_label_lines, _normalise
from PIL import Image

for img_id in [21, 26, 29, 33, 34, 24, 30, 31]:
    path = f'test_images/ocr_test_{img_id}_*.png'
    files = glob.glob(path)
    if not files:
        continue
    f = files[0]
    print(f'===== {os.path.basename(f)} =====')
    img = Image.open(f)
    print(f'  Image size: {img.size}')
    proc = preprocess_image(img)
    lines = _detect_lines(proc)
    print(f'  Lines found: {len(lines)}')
    # Print all lines
    for i, ln in enumerate(sorted(lines, key=lambda x: x['top'])):
        print(f'    Line {i}: y={ln["top"]} text="{_normalise(ln["text"])}"')
    
    labeled = _find_label_lines(lines)
    for key, anchor in labeled:
        print(f'  Label: {key} at y={anchor["top"]}, text="{anchor["text"]}"')
    
    result = extract_stats(f)
    print(f'  OCR Result: {json.dumps(result, indent=2)}')
    print()
