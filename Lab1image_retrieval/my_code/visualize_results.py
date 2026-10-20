
from pathlib import Path
import csv
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_DIR / "coil-100" / "coil-100"
RESULTS_DIR = PROJECT_DIR / "results"

csv_path = RESULTS_DIR / "top5_results.csv"

with open(csv_path, newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

query_name = rows[0]["query_image"]
query_rows = [
    row for row in rows if row["query_image"] == query_name
]
query_rows.sort(key=lambda row: int(row["rank"]))

thumb_size = 160
label_height = 42

canvas = Image.new(
    "RGB",
    (thumb_size * 3, (thumb_size + label_height) * 2),
    "white"
)
draw = ImageDraw.Draw(canvas)

items = [("QUERY: " + query_name, IMAGE_DIR / query_name)]

for row in query_rows:
    label = (
        f"Rank {row['rank']}: {row['retrieved_image']}\n"
        f"Relevant: {row['relevant']}"
    )
    items.append((label, IMAGE_DIR / row["retrieved_image"]))

for i, (label, path) in enumerate(items[:6]):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((thumb_size, thumb_size))

    x = (i % 3) * thumb_size
    y = (i // 3) * (thumb_size + label_height)

    canvas.paste(image, (x, y))
    draw.text((x + 3, y + thumb_size + 2), label, fill="black")

output_path = RESULTS_DIR / "top5_visualization.png"
canvas.save(output_path)

print("Saved visualization:", output_path)
print("Query image:", query_name)

for row in query_rows:
    print(
        f"Rank {row['rank']}: {row['retrieved_image']} | "
        f"Similarity: {float(row['cosine_similarity']):.4f} | "
        f"Relevant: {row['relevant']}"
    )
