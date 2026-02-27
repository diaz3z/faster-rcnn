import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F

try:
    import cv2
except ImportError:
    cv2 = None


def get_model(num_classes: int, pretrained_backbone: bool = False):
    # Supports both old and newer torchvision APIs.
    try:
        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            if pretrained_backbone
            else None
        )
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    except AttributeError:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained_backbone)

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


def draw_and_save_image(
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


def draw_boxes_on_frame(
    frame_bgr,
    prediction: Dict[str, torch.Tensor],
    class_names: List[str],
    threshold: float,
):
    boxes = prediction["boxes"].detach().cpu().numpy()
    labels = prediction["labels"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    kept = 0
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < threshold:
            continue

        x_min, y_min, x_max, y_max = box.astype(int)
        class_name = get_class_name(int(label), class_names)
        text = f"{class_name} ({score:.2f})"

        cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        text_y = max(y_min - 8, 10)
        cv2.putText(
            frame_bgr,
            text,
            (x_min, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        kept += 1

    return frame_bgr, kept


def run_video_inference(
    model,
    device: torch.device,
    video_path: str,
    video_output: str,
    class_names: List[str],
    threshold: float,
    show_plot: bool,
    save_output: bool,
):
    if cv2 is None:
        raise ImportError("OpenCV is required for video inference. Install it with: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    writer = None
    if save_output:
        output_path = Path(video_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    total_kept = 0

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            prediction = model(tensor)[0]

            annotated_frame, kept = draw_boxes_on_frame(
                frame_bgr=frame_bgr,
                prediction=prediction,
                class_names=class_names,
                threshold=threshold,
            )

            total_kept += kept
            frame_idx += 1

            if writer is not None:
                writer.write(annotated_frame)

            if show_plot:
                cv2.imshow("Video Inference", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames")

    cap.release()
    if writer is not None:
        writer.release()
    if show_plot:
        cv2.destroyAllWindows()

    if save_output:
        print(f"Video saved: {video_output}")
    print(f"Video processed. frames={frame_idx}, detections_above_threshold={total_kept}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained Faster R-CNN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--num-classes", type=int, default=4, help="Total classes including background")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["Background", "Chair", "Person", "Table"],
        help="Class names by index order (0..N-1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )
    parser.add_argument("--pretrained-backbone", action="store_true", help="Use pretrained backbone when building model")

    parser.add_argument("--image", type=str, default="", help="Single image path")
    parser.add_argument("--image-dir", type=str, default="", help="Directory containing images")

    parser.add_argument("--video", type=str, default="", help="Input video path")
    parser.add_argument(
        "--video-output",
        type=str,
        default="outputs/predictions/pred_video.mp4",
        help="Output path for annotated video",
    )

    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Directory for saved predictions")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--figsize", type=float, nargs=2, default=[12, 10], metavar=("W", "H"), help="Figure size")
    parser.add_argument("--show", action="store_true", help="Display prediction plots/video")
    parser.add_argument("--save", dest="save", action="store_true", help="Save prediction images/video")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Do not save prediction images/video")
    parser.set_defaults(save=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = get_model(num_classes=args.num_classes, pretrained_backbone=bool(args.pretrained_backbone))
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    threshold = float(args.score_threshold)
    class_names = args.class_names

    if args.video:
        run_video_inference(
            model=model,
            device=device,
            video_path=args.video,
            video_output=args.video_output,
            class_names=class_names,
            threshold=threshold,
            show_plot=bool(args.show),
            save_output=bool(args.save),
        )
        return

    if not args.image and not args.image_dir:
        raise ValueError("Provide at least one input: --image, --image-dir, or --video")

    image_paths = collect_images(args.image, args.image_dir)
    if not image_paths:
        raise ValueError("No input images found from --image/--image-dir")

    output_dir = Path(args.output_dir)
    show_plot = bool(args.show)
    save_plot = bool(args.save)
    figsize = args.figsize

    with torch.no_grad():
        for image_path in image_paths:
            image, image_tensor = prepare_image(image_path, device)
            prediction = model(image_tensor)[0]

            out_name = f"{image_path.stem}_pred{image_path.suffix}"
            output_path = output_dir / out_name
            kept = draw_and_save_image(
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
