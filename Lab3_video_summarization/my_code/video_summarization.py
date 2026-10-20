
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
VIDEO_PATH = BASE_DIR.parent / "Lab2_video_retrieval" / "dataset" / "video" / "WxtbjNsCQ8A.mp4"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Sample the video and select visually diverse frames.
SAMPLE_EVERY_SECONDS = 2
SUMMARY_SECONDS = 30


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if fps <= 0 or duration <= 0:
        cap.release()
        raise RuntimeError("Could not read video duration or frame rate.")

    sample_step = max(1, int(fps * SAMPLE_EVERY_SECONDS))
    samples = []
    positions = []

    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_no % sample_step == 0:
            small = cv2.resize(frame, (160, 90))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist(
                [hsv], [0, 1, 2], None,
                [8, 8, 8], [0, 180, 0, 256, 0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
            samples.append((frame_no, frame.copy(), hist))
            positions.append(frame_no / fps)

        frame_no += 1

    cap.release()

    if not samples:
        raise RuntimeError("No video frames were sampled.")

    # Choose samples with visually different histograms.
    # Keep the selected timestamps in chronological order.
    target_count = max(1, int(SUMMARY_SECONDS / SAMPLE_EVERY_SECONDS))
    selected = [0]
    remaining = list(range(1, len(samples)))

    while remaining and len(selected) < target_count:
        best_index = None
        best_distance = -1

        for candidate in remaining:
            distances = [
                cv2.compareHist(
                    samples[candidate][2].astype(np.float32),
                    samples[chosen][2].astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA
                )
                for chosen in selected
            ]
            distance = min(distances)

            if distance > best_distance:
                best_distance = distance
                best_index = candidate

        selected.append(best_index)
        remaining.remove(best_index)

    selected.sort(key=lambda i: samples[i][0])

    # Write a short summary video using the selected frames.
    height, width = samples[0][1].shape[:2]
    output_path = RESULTS_DIR / "summary_video.mp4"
    output_fps = 1.0 / SAMPLE_EVERY_SECONDS

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height)
    )

    frame_records = []
    for rank, index in enumerate(selected, start=1):
        frame_no, frame, _ = samples[index]
        writer.write(frame)

        timestamp = frame_no / fps
        frame_path = RESULTS_DIR / f"summary_frame_{rank:02d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        frame_records.append((rank, timestamp, frame_path.name))

    writer.release()

    summary_duration = len(selected) / output_fps
    proportion = min(summary_duration / duration, 1.0) * 100

    # Create a contact sheet of selected summary frames.
    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, index in zip(axes, selected):
        frame_no, frame, _ = samples[index]
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{frame_no / fps:.1f} seconds", fontsize=9)
        ax.axis("off")

    fig.suptitle("Selected Frames for Video Summary")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "summary_frames_contact_sheet.png", dpi=180)
    plt.close()

    # Save timestamps in a CSV file.
    with open(RESULTS_DIR / "summary_timestamps.csv", "w", encoding="utf-8") as file:
        file.write("frame_number,timestamp_seconds,frame_file\n")
        for rank, timestamp, filename in frame_records:
            file.write(f"{rank},{timestamp:.2f},{filename}\n")

    print("Video summarization completed.")
    print(f"Original duration: {duration:.2f} seconds")
    print(f"Selected frames: {len(selected)}")
    print(f"Summary duration: {summary_duration:.2f} seconds")
    print(f"Summary length: {proportion:.2f}% of original")
    print(f"Output video: {output_path}")
    print(f"Results folder: {RESULTS_DIR}")


if __name__ == "__main__":
    main()