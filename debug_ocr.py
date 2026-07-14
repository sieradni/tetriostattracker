import json
from pathlib import Path
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

img_path = Path("test_images") / "Screenshot 2026-07-13 005941.png"
print("Available:", is_available())

result = extract_stats(str(img_path))
print("Result:", json.dumps(result, indent=2))

# Also save preprocessed image and dump raw text
img = Image.open(img_path)
proc = preprocess_image(img)
proc.save("test_images/preprocessed.png")
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
