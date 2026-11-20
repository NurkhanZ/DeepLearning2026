import hashlib
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"
DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "african-wildlife"
OUTPUT_DIR = Path("results/dataset_samples")
CLASS_NAMES = {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}
COLORS = {0: (220, 50, 50), 1: (50, 140, 220), 2: (50, 180, 90), 3: (230, 150, 20)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def download_dataset():
    if (DATASET_DIR / "images/train").exists():
        print("Dataset already exists:", DATASET_DIR)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "african-wildlife.zip"
    if not zip_path.exists():
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATASET_DIR)


def remove_duplicates():
    seen_hashes = set()
    removed = {"train": 0, "val": 0, "test": 0}

    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            file_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
            if file_hash in seen_hashes:
                img_path.unlink()
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    lbl_path.unlink()
                removed[split] += 1
            else:
                seen_hashes.add(file_hash)

    return removed


def inspect_dataset(removed_duplicates):
    summary = {
        "removed_exact_duplicates": removed_duplicates,
        "splits": {},
        "issues": {},
        "cross_split_duplicates_after_cleaning": 0,
    }
    seen_hashes = {}
    cross_dups = 0

    for split in ["train", "val", "test"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

        class_counts = {name: 0 for name in CLASS_NAMES.values()}
        total_boxes = 0
        missing_labels = 0
        empty_labels = 0
        invalid_boxes = 0
        unreadable_images = 0

        for img_path in images:
            file_hash = hashlib.md5(img_path.read_bytes()).hexdigest()
            if file_hash in seen_hashes and seen_hashes[file_hash] != split:
                cross_dups += 1
            else:
                seen_hashes[file_hash] = split

            img = cv2.imread(str(img_path))
            if img is None:
                unreadable_images += 1
                continue

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                missing_labels += 1
                continue

            lines = [line.strip() for line in lbl_path.read_text().splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
                continue

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    invalid_boxes += 1
                    continue

                cls_id = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])
                valid_cls = cls_id in CLASS_NAMES
                valid_coords = 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                if not (valid_cls and valid_coords):
                    invalid_boxes += 1
                    continue

                class_counts[CLASS_NAMES[cls_id]] += 1
                total_boxes += 1

        summary["splits"][split] = {
            "images": len(images),
            "objects": total_boxes,
            "per_class": class_counts,
        }
        summary["issues"][split] = {
            "missing_labels": missing_labels,
            "empty_labels": empty_labels,
            "invalid_boxes": invalid_boxes,
            "unreadable_images": unreadable_images,
        }

    summary["cross_split_duplicates_after_cleaning"] = cross_dups
    return summary


def draw_ground_truth(img_path, lbl_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        color = COLORS[cls_id]
        label = CLASS_NAMES[cls_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return img


def save_sample_visualizations(seed=12345):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = DATASET_DIR / "images/train"
    lbl_dir = DATASET_DIR / "labels/train"

    rng = random.Random(seed)
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rng.shuffle(images)

    chosen = []
    seen_classes = set()
    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        lines = [line.strip() for line in lbl_path.read_text().splitlines() if line.strip()]
        if not lines:
            continue
        cls_id = int(float(lines[0].split()[0]))
        if cls_id not in seen_classes:
            chosen.append((img_path, lbl_path))
            seen_classes.add(cls_id)
        elif len(chosen) < 6 and len(lines) >= 2:
            chosen.append((img_path, lbl_path))
        if len(chosen) == 6:
            break

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for idx, (ax, (img_path, lbl_path)) in enumerate(zip(axes.flat, chosen), start=1):
        drawn = draw_ground_truth(img_path, lbl_path)
        plt.imsave(OUTPUT_DIR / f"gt_sample_{idx}.jpg", drawn)
        ax.imshow(drawn)
        ax.set_title(img_path.name, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ground_truth_grid.png", dpi=150)
    plt.close(fig)


def main():
    download_dataset()
    removed = remove_duplicates()
    summary = inspect_dataset(removed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    save_sample_visualizations()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
