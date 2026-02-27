import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
import torchvision
import yaml
from PIL import Image
from torchvision.transforms import functional as F


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model(num_classes: int, pretrained: bool = True):
    # Supports both old and newer torchvision APIs.
    try:
        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            if pretrained
            else None
        )
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    except AttributeError:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


def prepare_image(image_path: Path, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    return image, image_tensor


def collect_images(single_image: str, image_dir: str) -> List[Path]:
    images: List[Path] = []

    if single_image:
        images.append(Path(single_image))

    if image_dir:
        d = Path(image_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            images.extend(sorted(d.glob(ext)))

    # Keep order and remove duplicates.
    unique = []
    seen = set()
    for p in images:
        key = str(p.resolve())
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def get_class_name(class_id: int, class_names: List[str]) -> str:
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"Class_{class_id}"


def draw_and_save(
    image: Image.Image,
    prediction: Dict[str, torch.Tensor],
    class_names: List[str],
    threshold: float,
    figsize,
    output_path: Path,
    show_plot: bool,
    save_plot: bool,
):
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    plt.figure(figsize=figsize)
    plt.imshow(image)
    ax = plt.gca()

    kept = 0
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < threshold:
            continue

        x_min, y_min, x_max, y_max = box
        class_name = get_class_name(int(label), class_names)
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x_min, y_min, f"{class_name} ({score:.2f})", color="r")
        kept += 1

    plt.axis("off")

    if save_plot:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return kept


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained Faster R-CNN")
    parser.add_argument("--config", type=str, default="train.yaml", help="Path to YAML config")
    parser.add_argument("--image", type=str, default="", help="Optional single image override")
    parser.add_argument("--image-dir", type=str, default="", help="Optional directory override")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint override")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))

    device_cfg = str(cfg.get("device", "auto")).lower()
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    model_cfg = cfg["model"]
    infer_cfg = cfg.get("inference", {})

    checkpoint_path = args.checkpoint or infer_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("No checkpoint path provided. Set inference.checkpoint_path or use --checkpoint.")

    model = get_model(
        num_classes=int(model_cfg["num_classes"]),
        pretrained=bool(model_cfg.get("pretrained", True)),
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    image_path_cfg = infer_cfg.get("image_path", "")
    image_dir_cfg = infer_cfg.get("image_dir", "")
    image_paths = collect_images(args.image or image_path_cfg, args.image_dir or image_dir_cfg)
    if not image_paths:
        raise ValueError("No input images found. Set inference.image_path/image_dir or pass --image/--image-dir.")

    output_dir = Path(infer_cfg.get("output_dir", "outputs/predictions"))
    threshold = float(infer_cfg.get("score_threshold", 0.5))
    show_plot = bool(infer_cfg.get("show", True))
    save_plot = bool(infer_cfg.get("save", True))
    figsize = infer_cfg.get("figsize", [12, 10])
    class_names = model_cfg.get("class_names", [])

    with torch.no_grad():
        for image_path in image_paths:
            image, image_tensor = prepare_image(image_path, device)
            prediction = model(image_tensor)[0]

            out_name = f"{image_path.stem}_pred{image_path.suffix}"
            output_path = output_dir / out_name
            kept = draw_and_save(
                image=image,
                prediction=prediction,
                class_names=class_names,
                threshold=threshold,
                figsize=figsize,
                output_path=output_path,
                show_plot=show_plot,
                save_plot=save_plot,
            )
            print(f"{image_path} -> {output_path} | detections_above_threshold={kept}")


if __name__ == "__main__":
    main()
