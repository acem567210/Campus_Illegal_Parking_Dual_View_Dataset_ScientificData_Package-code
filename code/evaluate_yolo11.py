import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path("X:/parking_violation_yolo11_project")
DATA = ROOT / "data.yaml"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "runs" / "yolo11n_illegal_parking_e15-2" / "weights" / "best.pt"))
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="test_eval")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=str(DATA), split="test", imgsz=args.imgsz, device=args.device, project=str(ROOT / "runs"), name=args.name)
    print(metrics)


if __name__ == "__main__":
    main()
