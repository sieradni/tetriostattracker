"""Debug all_clears, timer, and PPS issues."""
import sys, json, glob
sys.path.insert(0, '.')
from ttr_tracker.ocr import (
    extract_stats, preprocess_image, _detect_lines, _find_label_lines,
    _find_all_clears_value, _focused_number_near_label, _parse_timer,
    _value_below, _parse_value_and_rate, _focused_rate_right, RE_PPS, RE_KPP, RE_SPP,
    _parse_rate, _fix_decimal
)
from PIL import Image

for img_id in [26, 28, 29, 31, 33, 34, 35]:
    path = f'test_images/ocr_test_{img_id}_*.png'
    files = glob.glob(path)
    if not files:
        continue
    f = files[0]
    print(f'===== {f.split("_")[-1]} =====')
    img = Image.open(f)
    proc = preprocess_image(img)
    lines = _detect_lines(proc)
    labeled = _find_label_lines(lines)
    
    # Check all_clears
    for key, anchor in labeled:
        if key == 'all_clears':
            print(f'  ALL CLEARS anchor: y={anchor["top"]}, text="{anchor["text"]}"')
            v1 = _find_all_clears_value(lines, anchor)
            print(f'    _find_all_clears_value: {v1}')
            v2 = _focused_number_near_label(img, anchor, preprocessed=proc)
            print(f'    _focused_number_near_label: {v2}')
    
    # Check timer
    timer = _parse_timer(lines, img.size[0], img.size[1])
    print(f'  Timer from _parse_timer: {timer}')
    
    # Check pieces value selection
    for key, anchor in labeled:
        if key == 'pieces_placed':
            val_line = _value_below(lines, anchor)
            print(f'  PIECES anchor y={anchor["top"]}, val_line: text="{val_line["text"] if val_line else None}" y={val_line["top"] if val_line else None}')
            if val_line:
                iv, pps, _ = _parse_value_and_rate(val_line)
                print(f'    parsed: int={iv}, pps={pps}')
                # Also check next line
                next_line = _value_below(lines, val_line, max_dist=60)
                if next_line:
                    print(f'    next_line: text="{next_line["text"]}" y={next_line["top"]}')
    
    # Check score reading
    for key, anchor in labeled:
        if key == 'score':
            mid_x = (anchor["left"] + anchor["right"]) / 2
            side = "left" if mid_x < img.size[0] * 0.6 else "right"
            val_line = _value_below(lines, anchor)
            print(f'  SCORE ({side}) anchor y={anchor["top"]}, val_line: text="{val_line["text"] if val_line else None}"')
    
    # Check SPP/rate
    print()
