
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "dataset"
VIDEO_DIR = DATA_DIR / "video"
INFO_FILE = DATA_DIR / "data" / "ydata-tvsum50-info.tsv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FRAMES_PER_VIDEO = 8
HIST_BINS = (8, 8, 8)


def load_metadata():
    """Load TVSum video titles and category labels."""
    metadata = {}

    with open(INFO_FILE, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            metadata[row["video_id"]] = {
                "category": row["category"],
                "title": row["title"],
            }

    return metadata


def extract_frame_features(video_path, number_of_frames=FRAMES_PER_VIDEO):
    """Sample frames and represent each with a normalized HSV histogram."""
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"WARNING: Could not open {video_path.name}")
        return [], []

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        capture.release()
        print(f"WARNING: No frames found in {video_path.name}")
        return [], []

    positions = np.linspace(
        0, max(total_frames - 1, 0),
        min(number_of_frames, total_frames),
        dtype=int,
    )

    features = []
    frames = []

    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        success, frame = capture.read()

        if not success:
            continue

        frame = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        histogram = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            list(HIST_BINS), [0, 180, 0, 256, 0, 256]
        )
        histogram = cv2.normalize(histogram, histogram).flatten()

        features.append(histogram)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    capture.release()
    return features, frames


def main():
    metadata = load_metadata()
    video_paths = sorted(VIDEO_DIR.glob("*.mp4"))

    if len(video_paths) < 4:
        raise RuntimeError(
            f"Expected at least 4 videos in {VIDEO_DIR}, "
            f"but found {len(video_paths)}."
        )

    print(f"Found {len(video_paths)} videos.")
    print("Extracting visual features. This may take a few minutes...")

    video_ids = []
    video_features = {}
    representative_frames = {}

    for index, path in enumerate(video_paths, start=1):
        video_id = path.stem
        features, frames = extract_frame_features(path)

        if not features:
            continue

        video_ids.append(video_id)
        video_features[video_id] = np.mean(features, axis=0)
        representative_frames[video_id] = frames[len(frames) // 2]

        print(f"[{index}/{len(video_paths)}] Processed {video_id}")

    if len(video_ids) < 4:
        raise RuntimeError("Too few videos could be processed.")

    feature_matrix = np.vstack([video_features[vid] for vid in video_ids])
    similarity_matrix = cosine_similarity(feature_matrix)
    similarity_matrix = np.clip(similarity_matrix, 0, 1)

    # Save the full video-to-video similarity matrix
    plt.figure(figsize=(11, 9))
    plt.imshow(similarity_matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Cosine similarity")
    plt.xticks(range(len(video_ids)), video_ids, rotation=90, fontsize=7)
    plt.yticks(range(len(video_ids)), video_ids, fontsize=7)
    plt.title("TVSum Video-to-Video Similarity Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "video_similarity_matrix.png", dpi=200)
    plt.close()

    # Use the first BK video as the default query.
    # You can change this ID to any of the 15 video IDs.
    query_id = "WxtbjNsCQ8A"

    if query_id not in video_ids:
        query_id = video_ids[0]

    query_index = video_ids.index(query_id)
    ranking = np.argsort(similarity_matrix[query_index])[::-1]
    ranking = [idx for idx in ranking if video_ids[idx] != query_id]
    top_three = ranking[:3]

    query_category = metadata.get(query_id, {}).get("category", "Unknown")
    relevant_count = 0
    rows = []

    print("\n" + "=" * 70)
    print(f"QUERY: {query_id}")
    print("Title:", metadata.get(query_id, {}).get("title", "Unknown"))
    print("Category:", query_category)
    print("\nTOP-3 MOST SIMILAR VIDEOS")
    print("=" * 70)

    for rank, idx in enumerate(top_three, start=1):
        candidate_id = video_ids[idx]
        candidate_category = metadata.get(candidate_id, {}).get(
            "category", "Unknown"
        )
        title = metadata.get(candidate_id, {}).get("title", "Unknown")
        score = float(similarity_matrix[query_index, idx])

        # Category agreement is the experiment's pseudo-relevance rule.
        relevant = candidate_category == query_category
        relevant_count += int(relevant)

        rows.append({
            "rank": rank,
            "query_id": query_id,
            "candidate_id": candidate_id,
            "title": title,
            "category": candidate_category,
            "similarity": f"{score:.4f}",
            "relevant_by_category": relevant,
        })

        print(f"{rank}. {candidate_id} | {title}")
        print(
            f"   Category: {candidate_category} | "
            f"Similarity: {score:.4f} | Relevant: {relevant}"
        )

    print(f"\nRelevant videos in Top-3: {relevant_count}/3")
    print("Note: relevance is approximated by matching TVSum categories.")

    # Save Top-3 results as a CSV table
    csv_path = RESULTS_DIR / "top3_results.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Show representative frames for the query and its Top-3 candidates
    selected_ids = [query_id] + [video_ids[idx] for idx in top_three]
    fig, axes = plt.subplots(1, len(selected_ids), figsize=(16, 4))

    for axis, vid in zip(axes, selected_ids):
        axis.imshow(representative_frames[vid])
        title = metadata.get(vid, {}).get("title", vid)
        axis.set_title(f"{vid}\n{title[:35]}", fontsize=8)
        axis.axis("off")

    fig.suptitle("Query Video and Top-3 Retrieval Results")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top3_representative_frames.png", dpi=200)
    plt.close()

    # Frame-to-frame similarity matrix for the query and top-ranked video
    if top_three:
        candidate_id = video_ids[top_three[0]]
        query_features, _ = extract_frame_features(VIDEO_DIR / f"{query_id}.mp4")
        candidate_features, _ = extract_frame_features(
            VIDEO_DIR / f"{candidate_id}.mp4"
        )

        frame_matrix = cosine_similarity(
            np.vstack(query_features),
            np.vstack(candidate_features),
        )
        frame_matrix = np.clip(frame_matrix, 0, 1)

        plt.figure(figsize=(7, 5))
        plt.imshow(frame_matrix, cmap="magma", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(label="Frame similarity")
        plt.xlabel(f"Frames from candidate {candidate_id}")
        plt.ylabel(f"Frames from query {query_id}")
        plt.title("Frame-to-Frame Similarity")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "frame_similarity_matrix.png", dpi=200)
        plt.close()

    print("\nSaved results to:", RESULTS_DIR)
    print("- top3_results.csv")
    print("- video_similarity_matrix.png")
    print("- top3_representative_frames.png")
    print("- frame_similarity_matrix.png")


if __name__ == "__main__":
    main()