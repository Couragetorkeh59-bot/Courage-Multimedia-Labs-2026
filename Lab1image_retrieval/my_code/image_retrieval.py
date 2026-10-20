
from pathlib import Path
from PIL import Image
import numpy as np
import csv

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_DIR / "coil-100" / "coil-100"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load image paths
image_paths = sorted(IMAGE_DIR.glob("*.png"))

def get_object_id(path):
    return path.stem.split("__")[0]

def color_feature(path):
    """Create a normalized RGB color histogram."""
    with Image.open(path) as image:
        image = image.convert("RGB").resize((64, 64))
        pixels = np.asarray(image)

    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(
            pixels[:, :, channel],
            bins=16,
            range=(0, 256)
        )
        histograms.extend(hist)

    feature = np.asarray(histograms, dtype=np.float32)
    norm = np.linalg.norm(feature)
    return feature / norm if norm else feature

print("Extracting image features...")
features = np.stack([color_feature(path) for path in image_paths])
print(f"Images processed: {len(image_paths)}")

# Use every 10th image as a query; other images form the gallery.
query_indices = list(range(0, len(image_paths), 10))
query_set = set(query_indices)
gallery_indices = [
    i for i in range(len(image_paths)) if i not in query_set
]

rows = []
precision_values = []

print("Running Top-5 retrieval...")
for query_index in query_indices:
    query_feature = features[query_index]
    gallery_features = features[gallery_indices]

    similarities = gallery_features @ query_feature
    top_positions = np.argsort(similarities)[::-1][:5]
    top_indices = [gallery_indices[pos] for pos in top_positions]

    query_object = get_object_id(image_paths[query_index])
    relevant_count = sum(
        get_object_id(image_paths[i]) == query_object
        for i in top_indices
    )
    precision_at_5 = relevant_count / 5
    precision_values.append(precision_at_5)

    for rank, (index, position) in enumerate(
        zip(top_indices, top_positions), start=1
    ):
        rows.append({
            "query_image": image_paths[query_index].name,
            "query_object_id": query_object,
            "rank": rank,
            "retrieved_image": image_paths[index].name,
            "retrieved_object_id": get_object_id(image_paths[index]),
            "cosine_similarity": float(similarities[position]),
            "relevant": int(
                get_object_id(image_paths[index]) == query_object
            )
        })

# Save detailed retrieval results
csv_path = RESULTS_DIR / "top5_results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

mean_precision = float(np.mean(precision_values))

summary_path = RESULTS_DIR / "evaluation_summary.txt"
with open(summary_path, "w", encoding="utf-8") as file:
    file.write(f"Dataset images: {len(image_paths)}\n")
    file.write(f"Query count: {len(query_indices)}\n")
    file.write(f"Gallery count: {len(gallery_indices)}\n")
    file.write("Feature method: normalized RGB color histogram\n")
    file.write("Similarity: cosine similarity\n")
    file.write("Relevance: same COIL-100 object ID\n")
    file.write(f"Mean Precision@5: {mean_precision:.4f}\n")

print(f"Query count: {len(query_indices)}")
print(f"Gallery count: {len(gallery_indices)}")
print(f"Mean Precision@5: {mean_precision:.4f}")
print(f"Detailed results saved to: {csv_path}")
print(f"Evaluation summary saved to: {summary_path}")
