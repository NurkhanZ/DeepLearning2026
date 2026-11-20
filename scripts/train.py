import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO


def resolve_dataset_yaml(yaml_path, out_path):
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    cfg["path"] = str((Path(yaml_path).parent / cfg["path"]).resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_name = cfg["run_name"]
    out_dir = Path("results") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path("runs") / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)

    resolved_data = resolve_dataset_yaml(cfg["data"], Path("runs") / f"{run_name}_dataset.yaml")

    model = YOLO(cfg["model"])
    start_time = time.time()
    model.train(
        data=resolved_data,
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        lr0=cfg["lr0"],
        seed=cfg["seed"],
        workers=cfg["workers"],
        project="runs",
        name=run_name,
        exist_ok=True,
        deterministic=True,
        plots=True,
    )
    train_seconds = round(time.time() - start_time, 1)

    run_dir = Path("runs") / run_name
    best_weights = run_dir / "weights" / "best.pt"

    best_model = YOLO(str(best_weights))
    val_res = best_model.val(data=resolved_data, split="val", imgsz=cfg["imgsz"], plots=False)

    shutil.copy2(args.config, out_dir / "config.yaml")
    for fname in ["results.csv", "results.png", "PR_curve.png", "confusion_matrix.png"]:
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    metrics = {
        "run_name": run_name,
        "framework": f"ultralytics=={ultralytics.__version__}",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model": cfg["model"],
        "imgsz": cfg["imgsz"],
        "epochs": cfg["epochs"],
        "batch": cfg["batch"],
        "lr0": cfg["lr0"],
        "seed": cfg["seed"],
        "training_time_sec": train_seconds,
        "best_weights": str(best_weights),
        "val_precision": round(float(val_res.box.mp), 4),
        "val_recall": round(float(val_res.box.mr), 4),
        "val_map50": round(float(val_res.box.map50), 4),
        "val_map50_95": round(float(val_res.box.map), 4),
        "val_per_class_map50": {
            val_res.names[i]: round(float(ap), 4) for i, ap in enumerate(val_res.box.ap50)
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

