import sys
import json
from pathlib import Path

# Ensure project root is on sys.path when run from scripts/
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ttr_tracker.ocr import extract_stats, is_available, preprocess_image
import pytesseract

# Ensure pytesseract can find tesseract
_tess_paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
try:
    pytesseract.get_tesseract_version()
except Exception:
    for p in _tess_paths:
        if Path(p).exists():
            pytesseract.pytesseract.tesseract_cmd = p
            break

from PIL import Image

img_path = Path("tests") / "test_images" / "Screenshot 2026-07-13 005941.png"
print("Available:", is_available())

result = extract_stats(str(img_path))
print("Result:", json.dumps(result, indent=2))

# Also save preprocessed image and dump raw text
img = Image.open(img_path)
proc = preprocess_image(img)
proc.save("tests/test_images/preprocessed.png")
print("Saved preprocessed image")

data = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT, config="--psm 11 --oem 3")
lines = {}
for i in range(len(data["text"])):
    txt = data["text"][i].strip()
    if not txt:
        continue
    key = (data["block_num"][i], data["line_num"][i])
    conf = data["conf"][i]
    x, y = data["left"][i], data["top"][i]
    if key not in lines:
        lines[key] = {"text": txt, "conf": conf, "x": x, "y": y}
    else:
        lines[key]["text"] += " " + txt
        lines[key]["conf"] = max(lines[key]["conf"], conf)

print("\nRaw detected text:")
for k, v in sorted(lines.items(), key=lambda kv: (kv[1]["y"], kv[1]["x"])):
    print('  pos=({:4},{:4}) conf={:3}  text="{}"'.format(v["x"], v["y"], v["conf"], v["text"]))
