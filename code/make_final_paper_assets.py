import csv
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "parking_violation_yolo11_project"
ASSETS = PROJECT / "paper_assets"
PRED = PROJECT / "outputs" / "final_test_predictions"


DATASET_TABLE = [
    ["Total images", 584],
    ["Vehicle groups", 277],
    ["Front-view images", 287],
    ["Top-view images", 297],
    ["Train images", 405],
    ["Validation images", 88],
    ["Test images", 91],
]

RESULTS_TABLE = [
    ["YOLO11n-old-auto-label", "test", 0.9555, 0.9433, 0.9686, 0.7650, 17.5],
    ["YOLO11n-final-label-img320", "test", 0.9890, 0.9848, 0.9846, 0.8361, 7.1],
    ["YOLO11n-final-label", "val", 0.9420, 0.9550, 0.9510, 0.8320, 16.9],
    ["YOLO11n-final-label", "test", 0.9861, 0.9890, 0.9864, 0.8726, 17.6],
    ["YOLO11n-final-label-img512", "test", 0.9782, 0.9868, 0.9859, 0.8618, 18.4],
    ["YOLO11s-final-label", "test", 0.9889, 0.9777, 0.9880, 0.8605, 31.2],
    ["YOLO11n-final-label", "test-front", 0.9770, 0.9790, 0.9580, 0.5400, 19.7],
    ["YOLO11n-final-label", "test-top", 0.9970, 1.0000, 0.9950, 0.7960, 16.4],
]

IMGSZ_TABLE = [
    ["YOLO11n", 320, 0.9890, 0.9848, 0.9846, 0.8361, 7.1],
    ["YOLO11n", 416, 0.9861, 0.9890, 0.9864, 0.8726, 17.6],
    ["YOLO11n", 512, 0.9782, 0.9868, 0.9859, 0.8618, 18.4],
]

MODEL_SCALE_TABLE = [
    ["YOLO11n", 2.58, 6.3, 0.9861, 0.9890, 0.9864, 0.8726, 17.6],
    ["YOLO11s", 9.41, 21.3, 0.9889, 0.9777, 0.9880, 0.8605, 31.2],
]


def write_csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def bar_chart(items, title, out_path, max_value=None):
    w, h = 980, 560
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 28), title, fill=(20, 20, 20))
    left, top, bar_h, gap = 240, 92, 44, 24
    max_v = max_value or max(v for _, v in items)
    colors = [(42, 110, 187), (36, 150, 115), (229, 134, 6), (123, 97, 255), (210, 81, 94)]
    for i, (label, value) in enumerate(items):
        y = top + i * (bar_h + gap)
        bw = int((w - 340) * value / max_v)
        d.text((40, y + 13), str(label), fill=(0, 0, 0))
        d.rectangle([left, y, left + bw, y + bar_h], fill=colors[i % len(colors)])
        d.text((left + bw + 10, y + 13), f"{value:.4f}" if value <= 1 else str(value), fill=(0, 0, 0))
    img.save(out_path)


def make_prediction_montage():
    files = sorted(PRED.glob("*.*"))[:16]
    cells = []
    for p in files:
        im = Image.open(p).convert("RGB")
        im.thumbnail((240, 180))
        cell = Image.new("RGB", (260, 216), "white")
        cell.paste(im, ((260 - im.width) // 2, 0))
        d = ImageDraw.Draw(cell)
        d.text((6, 186), p.name[:32], fill=(0, 0, 0))
        cells.append(cell)
    out = Image.new("RGB", (260 * 4, 216 * 4), "white")
    for i, cell in enumerate(cells):
        out.paste(cell, ((i % 4) * 260, (i // 4) * 216))
    out.save(ASSETS / "fig_prediction_examples.jpg")


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    write_csv(ASSETS / "table_dataset.csv", ["Item", "Count"], DATASET_TABLE)
    write_csv(
        ASSETS / "table_results.csv",
        ["Model", "Split", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "Inference ms/img"],
        RESULTS_TABLE,
    )
    write_csv(
        ASSETS / "table_imgsz_ablation.csv",
        ["Model", "Input size", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "Inference ms/img"],
        IMGSZ_TABLE,
    )
    write_csv(
        ASSETS / "table_model_scale.csv",
        ["Model", "Params(M)", "GFLOPs", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "Inference ms/img"],
        MODEL_SCALE_TABLE,
    )

    bar_chart([("train", 405), ("val", 88), ("test", 91)], "Dataset Split", ASSETS / "fig_split_distribution.png")
    bar_chart([("front", 287), ("top", 297)], "Viewpoint Distribution", ASSETS / "fig_view_distribution.png")
    bar_chart(
        [("old auto-label", 0.7650), ("final label", 0.8726)],
        "Ablation: Label Strategy on Test mAP@0.5:0.95",
        ASSETS / "fig_ablation_map95.png",
        max_value=1.0,
    )
    bar_chart(
        [("320", 0.8361), ("416", 0.8726), ("512", 0.8618)],
        "Input-size Ablation: Test mAP@0.5:0.95",
        ASSETS / "fig_imgsz_ablation.png",
        max_value=1.0,
    )
    bar_chart(
        [("YOLO11n", 0.8726), ("YOLO11s", 0.8605)],
        "Model-scale Comparison: Test mAP@0.5:0.95",
        ASSETS / "fig_model_scale_comparison.png",
        max_value=1.0,
    )
    bar_chart(
        [("overall", 0.8726), ("front", 0.5400), ("top", 0.7960)],
        "View-wise Test mAP@0.5:0.95",
        ASSETS / "fig_view_map95.png",
        max_value=1.0,
    )
    make_prediction_montage()

    key_plots = [
        PROJECT / "runs" / "yolo11n_hybrid_final_e20" / "results.png",
        PROJECT / "runs" / "test_eval_hybrid_final_e20" / "BoxPR_curve.png",
        PROJECT / "runs" / "test_eval_hybrid_final_e20" / "confusion_matrix.png",
    ]
    for p in key_plots:
        if p.exists():
            shutil.copy2(p, ASSETS / p.name)
    print(ASSETS)


if __name__ == "__main__":
    main()
