import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

SOURCE = Path("dataset/gemma_edited_frame_consistency.jsonl")
OUTPUT = Path("dataset/gemma_frame_audit_sample/comparisons_30")
SEED = 20260826
PER_CLASS = 15


def font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def samples():
    rng = random.Random(SEED)
    buckets = {True: [], False: []}
    seen = {True: 0, False: 0}
    with SOURCE.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            passed = row["verdict"]["pass"]
            seen[passed] += 1
            if len(buckets[passed]) < PER_CLASS:
                buckets[passed].append(row)
            else:
                position = rng.randrange(seen[passed])
                if position < PER_CLASS:
                    buckets[passed][position] = row
    return buckets[True] + buckets[False]


def load_image(path, size):
    image = Image.open(path).convert("RGB")
    return ImageOps.contain(image, size, Image.Resampling.LANCZOS)


def comparison(row, index):
    passed = row["verdict"]["pass"]
    label = "PASS" if passed else "FAIL"
    color = "#16834b" if passed else "#cf303b"
    panel_w, panel_h = 900, 900
    header_h, section_h, footer_h = 130, 70, 90
    image_y = header_h + section_h
    canvas = Image.new("RGB", (panel_w * 2, image_y + panel_h + footer_h), "#161616")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, panel_w * 2, header_h), fill=color)
    draw.text((35, 28), f"#{index:02d}  GEMMA: {label}", font=font(60), fill="white")
    draw.rectangle((0, header_h, panel_w * 2, image_y), fill="#303030")
    draw.text((35, header_h + 10), "INPUT", font=font(42), fill="white")
    draw.text((panel_w + 35, header_h + 10), "OUTPUT / EDITED", font=font(42), fill="white")
    for column, key in enumerate(("model_input_path", "edited_path")):
        try:
            image = load_image(row[key], (panel_w - 30, panel_h - 30))
            x = column * panel_w + (panel_w - image.width) // 2
            y = image_y + (panel_h - image.height) // 2
            canvas.paste(image, (x, y))
        except Exception as exc:
            draw.text(
                (column * panel_w + 35, image_y + 100),
                f"IMAGE LOAD ERROR: {type(exc).__name__}",
                font=font(30),
                fill="white",
            )
    draw.line((panel_w, header_h, panel_w, image_y + panel_h), fill="white", width=5)
    info = f"frame={row['frame_index']}   object={row['object_name']}   sample={row['sample_id']}"
    draw.text((35, image_y + panel_h + 24), info, font=font(28), fill="white")
    return canvas


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = samples()
    manifest = OUTPUT.parent / "samples.jsonl"
    with manifest.open("w", encoding="utf-8") as output:
        for index, row in enumerate(selected, 1):
            label = "pass" if row["verdict"]["pass"] else "fail"
            comparison(row, index).save(
                OUTPUT / f"{index:02d}_{label}.jpg", quality=96, subsampling=0
            )
            record = {
                "display_index": index,
                "pass": row["verdict"]["pass"],
                "sample_id": row["sample_id"],
                "frame_index": row["frame_index"],
                "object_name": row["object_name"],
                "model_input_path": row["model_input_path"],
                "edited_path": row["edited_path"],
                "defects": row["verdict"]["defects"],
                "reason": row["verdict"]["reason"],
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
