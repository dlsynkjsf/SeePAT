from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def build_review_sheets(
    input_dir: Path,
    output_dir: Path,
    checklist_path: Path | None = None,
    columns: int = 5,
    rows: int = 5,
) -> list[Path]:
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as error:
        raise RuntimeError(
            'Install preprocessing dependencies with: pip install -e ".[preprocess]"'
        ) from error

    images = sorted(input_dir.glob("*_minimum.png"))
    if not images:
        raise FileNotFoundError(f"No minimum-closure overlays found under {input_dir}")
    if columns < 1 or rows < 1:
        raise ValueError("Contact-sheet rows and columns must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    cell_width = 224
    image_height = 224
    label_height = 20
    cell_height = image_height + label_height
    per_sheet = rows * columns
    sheets: list[Path] = []

    for sheet_index in range(math.ceil(len(images) / per_sheet)):
        selected = images[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        for index, image_path in enumerate(selected):
            with Image.open(image_path) as source:
                thumbnail = ImageOps.fit(source.convert("RGB"), (cell_width, image_height))
            x = (index % columns) * cell_width
            y = (index // columns) * cell_height
            canvas.paste(thumbnail, (x, y))
            draw.rectangle(
                (x, y + image_height, x + cell_width, y + cell_height), fill="white"
            )
            draw.text((x + 4, y + image_height + 3), image_path.stem, fill="black")

        output_path = output_dir / f"review_{sheet_index + 1:02d}.jpg"
        canvas.save(output_path, quality=92)
        sheets.append(output_path)

    checklist_path = checklist_path or output_dir / "review_checklist.csv"
    checklist_path.parent.mkdir(parents=True, exist_ok=True)
    existing_reviews: dict[str, dict[str, str]] = {}
    if checklist_path.is_file():
        with checklist_path.open(newline="", encoding="utf-8") as stream:
            existing_reviews = {
                str(row.get("event_id", "")): row for row in csv.DictReader(stream)
            }
    with checklist_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "event_id",
                "overlay_path",
                "reviewer",
                "review_status",
                "notes",
            ),
        )
        writer.writeheader()
        for path in images:
            event_id = path.stem.removesuffix("_minimum")
            previous = existing_reviews.get(event_id, {})
            writer.writerow(
                {
                    "event_id": event_id,
                    "overlay_path": str(path),
                    "reviewer": previous.get("reviewer", ""),
                    "review_status": previous.get("review_status", "pending"),
                    "notes": previous.get("notes", ""),
                }
            )
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.qa")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checklist", type=Path)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    sheets = build_review_sheets(
        args.input_dir,
        args.output_dir,
        checklist_path=args.checklist,
        columns=args.columns,
        rows=args.rows,
    )
    print(f"Wrote {len(sheets)} review sheets to {args.output_dir}")


if __name__ == "__main__":
    main()
