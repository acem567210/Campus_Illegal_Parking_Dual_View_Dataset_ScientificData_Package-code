import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path("X:/parking_violation_yolo11_project")
DATA = ROOT / "data.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="yolo11n_illegal_parking_e15")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(DATA),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
        patience=8,
        seed=42,
        workers=0,
        pretrained=True,
        cache=False,
    )


if __name__ == "__main__":
    main()
