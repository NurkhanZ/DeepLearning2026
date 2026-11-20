import argparse
import json
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

CLASS_NAMES = {0: "buffalo", 1: "elephant", 2: "rhino", 3: "zebra"}
COLORS = {0: (220, 50, 50), 1: (50, 140, 220), 2: (50, 180, 90), 3: (230, 150, 20)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def resolve_dataset_yaml(yaml_path, out_path):
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    cfg["path"] = str((Path(yaml_path).parent / cfg["path"]).resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out_path), Path(cfg["path"])


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes(lbl_path, w, h):
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:])
        boxes.append((cls_id, [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]))
    return boxes


def draw_boxes(img_rgb, boxes, show_conf=False):
    canvas = img_rgb.copy()
    for item in boxes:
        if show_conf:
            cls_id, coords, conf = item
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        else:
            cls_id, coords = item
            label = CLASS_NAMES[cls_id]
        x1, y1, x2, y2 = map(int, coords)
        color = COLORS[cls_id]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas


def analyze_image(gt_boxes, pred_boxes, iou_thr=0.5):
    matched_gt = set()
    false_positives = 0
    wrong_class = 0

    for p_cls, p_box, _ in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, (g_cls, g_box) in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = box_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_thr and best_idx != -1:
            matched_gt.add(best_idx)
            if gt_boxes[best_idx][0] != p_cls:
                wrong_class += 1
        else:
            false_positives += 1

    missed = len(gt_boxes) - len(matched_gt)
    return missed, false_positives, wrong_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()

    out_dir = Path("results/test_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_data, dataset_root = resolve_dataset_yaml(args.data, Path("runs/test_dataset.yaml"))

    model = YOLO(args.weights)
    val_res = model.val(
        data=resolved_data,
        split="test",
        imgsz=args.imgsz,
        iou=args.iou,
        project="runs",
        name="test_eval",
        exist_ok=True,
        plots=True,
    )

    eval_dir = Path("runs/test_eval")
    for fname in ["PR_curve.png", "confusion_matrix.png"]:
        src = eval_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    test_metrics = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "prediction_conf_threshold": args.conf,
        "nms_iou_threshold": args.iou,
        "eval_iou_threshold": 0.50,
        "test_precision": round(float(val_res.box.mp), 4),
        "test_recall": round(float(val_res.box.mr), 4),
        "test_map50": round(float(val_res.box.map50), 4),
        "test_map50_95": round(float(val_res.box.map), 4),
        "test_per_class_map50": {
            val_res.names[i]: round(float(ap), 4) for i, ap in enumerate(val_res.box.ap50)
        },
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    img_dir = dataset_root / "images/test"
    lbl_dir = dataset_root / "labels/test"
    test_images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    per_class_good = {}
    extra_good = []
    failures = []

    for img_path in test_images:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        gt_boxes = load_gt_boxes(lbl_dir / f"{img_path.stem}.txt", w, h)

        pred = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
        pred_boxes = []
        for box in pred.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            coords = box.xyxy[0].tolist()
            pred_boxes.append((cls_id, coords, conf))

        missed, fp, wrong_cls = analyze_image(gt_boxes, pred_boxes, iou_thr=0.50)
        total_err = missed + fp + wrong_cls

        if total_err == 0 and pred_boxes:
            first_cls = pred_boxes[0][0]
            if first_cls not in per_class_good:
                per_class_good[first_cls] = (img_path, img_rgb, pred_boxes)
            elif len(pred_boxes) >= 2 and len(extra_good) < 4:
                extra_good.append((img_path, img_rgb, pred_boxes))
        elif total_err > 0:
            failures.append({
                "image": img_path.name,
                "path": img_path,
                "rgb": img_rgb,
                "gt": gt_boxes,
                "pred": pred_boxes,
                "missed": missed,
                "false_positives": fp,
                "wrong_class": wrong_cls,
                "total_errors": total_err,
            })

    good_samples = [per_class_good[c] for c in sorted(per_class_good)] + extra_good

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for idx, (ax, (img_path, img_rgb, pred_boxes)) in enumerate(zip(axes.flat, good_samples[:6]), start=1):
        drawn = draw_boxes(img_rgb, pred_boxes, show_conf=True)
        plt.imsave(out_dir / f"pred_{idx}.jpg", drawn)
        ax.imshow(drawn)
        ax.set_title(img_path.name, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "test_predictions_grid.png", dpi=150)
    plt.close(fig)

    failures.sort(key=lambda x: (-x["wrong_class"], -x["total_errors"], x["image"]))
    selected_failures = []
    for kind in ["wrong_class", "missed", "false_positives"]:
        for item in failures:
            if item[kind] > 0 and item not in selected_failures:
                selected_failures.append(item)
                break
    for item in failures:
        if len(selected_failures) >= 3:
            break
        if item not in selected_failures:
            selected_failures.append(item)

    failure_report = []
    for idx, item in enumerate(selected_failures[:3], start=1):
        gt_img = draw_boxes(item["rgb"], item["gt"], show_conf=False)
        pred_img = draw_boxes(item["rgb"], item["pred"], show_conf=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(gt_img)
        axes[0].set_title(f"Ground Truth ({item['image']})", fontsize=11)
        axes[0].axis("off")
        axes[1].imshow(pred_img)
        axes[1].set_title(
            f"Prediction (missed={item['missed']}, fp={item['false_positives']}, wrong_cls={item['wrong_class']})",
            fontsize=11,
        )
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"failure_{idx}.png", dpi=150)
        plt.close(fig)

        failure_report.append({
            "case": idx,
            "image": item["image"],
            "missed": item["missed"],
            "false_positives": item["false_positives"],
            "wrong_class": item["wrong_class"],
            "gt_classes": [CLASS_NAMES[c] for c, _ in item["gt"]],
            "pred_classes": [f"{CLASS_NAMES[c]} ({conf:.2f})" for c, _, conf in item["pred"]],
        })

    (out_dir / "failure_cases.json").write_text(json.dumps(failure_report, indent=2))
    print(json.dumps(test_metrics, indent=2))
    print("Failure cases:", json.dumps(failure_report, indent=2))


if __name__ == "__main__":
    main()

