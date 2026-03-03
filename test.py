import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

    unique: List[Path] = []
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


def unwrap_checkpoint(obj: Any) -> Tuple[dict, dict]:
    """
    Returns state_dict, meta

    Supported formats:
      1) pure state_dict
      2) {"model": state_dict, ...}
      3) {"state_dict": state_dict, ...}
      4) {"model_state_dict": state_dict, ...}
      5) {"model_state": state_dict, ...}   new
    """
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint must be a dict or a state_dict-like dict")

    def looks_like_state_dict(d: dict) -> bool:
        if not d:
            return False
        k = next(iter(d.keys()))
        return isinstance(k, str) and (
            "roi_heads" in k or "backbone" in k or "rpn" in k or "transform" in k
        )

    if looks_like_state_dict(obj):
        return obj, {}

    for k in ("model", "state_dict", "model_state_dict", "model_state"):
        v = obj.get(k)
        if isinstance(v, dict) and looks_like_state_dict(v):
            return v, {kk: vv for kk, vv in obj.items() if kk != k}

    # If it is a dict but not a known wrapper, treat it as meta not state_dict
    raise KeyError(
        "Checkpoint dict did not contain a model state under keys model, state_dict, "
        "model_state_dict, or model_state"
    )


# def unwrap_checkpoint(obj: Any) -> Tuple[dict, dict]:
#     """
#     Returns state_dict, meta

#     Supported formats:
#       1) pure state_dict
#       2) {"model": state_dict, ...}
#       3) {"state_dict": state_dict, ...}
#       4) {"model_state_dict": state_dict, ...}
#     meta contains everything else (potentially including class names if you saved them).
#     """
#     if not isinstance(obj, dict):
#         raise TypeError("Checkpoint must be a dict or a state_dict-like dict")

#     def looks_like_state_dict(d: dict) -> bool:
#         if not d:
#             return False
#         k = next(iter(d.keys()))
#         return isinstance(k, str) and (
#             "roi_heads" in k or "backbone" in k or "rpn" in k or "transform" in k
#         )

#     if looks_like_state_dict(obj):
#         return obj, {}

#     for k in ("model", "state_dict", "model_state_dict"):
#         v = obj.get(k)
#         if isinstance(v, dict):
#             return v, {kk: vv for kk, vv in obj.items() if kk != k}

#     # Fallback: treat entire dict as state_dict
#     return obj, {}


def infer_num_classes_from_state_dict(state_dict: dict) -> int:
    key = "roi_heads.box_predictor.cls_score.weight"
    if key not in state_dict:
        raise KeyError(
            f"Could not find {key} in checkpoint. Keys example: {list(state_dict.keys())[:10]}"
        )
    # weight shape [num_classes, 1024]
    return int(state_dict[key].shape[0])


def extract_class_names_from_meta(meta: dict) -> Optional[List[str]]:
    """
    Works only if you saved names in the checkpoint dict during training.
    Common keys: class_names, classes, categories, labels, label_map
    """
    for k in ("class_names", "classes", "categories", "labels"):
        v = meta.get(k)
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v

    lm = meta.get("label_map") or meta.get("labelmap")
    if isinstance(lm, dict) and lm:
        # id -> name
        if all(isinstance(k, int) for k in lm.keys()) and all(isinstance(v, str) for v in lm.values()):
            return [lm[i] for i in sorted(lm.keys())]
        # name -> id
        if all(isinstance(k, str) for k in lm.keys()) and all(isinstance(v, int) for v in lm.values()):
            inv = {v: k for k, v in lm.items()}
            return [inv[i] for i in sorted(inv.keys())]

    return None


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
        raise ImportError("OpenCV is required for video inference. Install it with pip install opencv-python")

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
            tensor = (
                torch.from_numpy(frame_rgb)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(device)
            )
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


def run_webcam_inference(
    model,
    device: torch.device,
    webcam_index: int,
    webcam_output: str,
    class_names: List[str],
    threshold: float,
    show_plot: bool,
    save_output: bool,
):
    if cv2 is None:
        raise ImportError("OpenCV is required for webcam inference. Install it with pip install opencv-python")

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        raise ValueError(f"Could not open webcam index: {webcam_index}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    writer = None
    if save_output:
        output_path = Path(webcam_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    total_kept = 0

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Webcam frame read failed. Exiting.")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(frame_rgb)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(device)
            )
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
                cv2.imshow("Webcam Inference press q to quit", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % 60 == 0:
                print(f"Processed {frame_idx} frames")

    cap.release()
    if writer is not None:
        writer.release()
    if show_plot:
        cv2.destroyAllWindows()

    if save_output:
        print(f"Webcam video saved: {webcam_output}")
    print(f"Webcam processed. frames={frame_idx}, detections_above_threshold={total_kept}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained Faster R-CNN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")

    # Keep args, but they will be overridden by checkpoint if mismatched
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

    parser.add_argument("--webcam", action="store_true", help="Run inference on webcam feed")
    parser.add_argument("--webcam-index", type=int, default=0, help="Webcam device index")
    parser.add_argument(
        "--webcam-output",
        type=str,
        default="outputs/predictions/pred_webcam.mp4",
        help="Output path for annotated webcam video",
    )

    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Directory for saved predictions")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--figsize", type=float, nargs=2, default=[12, 10], metavar=("W", "H"), help="Figure size")
    parser.add_argument("--show", action="store_true", help="Display prediction plots or video windows")
    parser.add_argument("--save", dest="save", action="store_true", help="Save prediction images or video")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Do not save prediction images or video")
    parser.set_defaults(save=True)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load checkpoint first so we can auto-infer num_classes (and maybe class names)
    ckpt_obj = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict, meta = unwrap_checkpoint(ckpt_obj)

    ckpt_num_classes = infer_num_classes_from_state_dict(state_dict)
    if args.num_classes != ckpt_num_classes:
        print(f"Checkpoint classes={ckpt_num_classes}. Overriding --num-classes {args.num_classes} -> {ckpt_num_classes}")
        args.num_classes = ckpt_num_classes

    ckpt_class_names = extract_class_names_from_meta(meta)
    if ckpt_class_names and len(ckpt_class_names) == args.num_classes:
        args.class_names = ckpt_class_names
        print(f"Loaded class names from checkpoint: {args.class_names}")
    else:
        if ckpt_class_names and len(ckpt_class_names) != args.num_classes:
            print(
                f"Checkpoint provided class names length={len(ckpt_class_names)} but num_classes={args.num_classes}. "
                "Ignoring checkpoint class names."
            )

        if len(args.class_names) != args.num_classes:
            print(
                f"--class-names length={len(args.class_names)} does not match num_classes={args.num_classes}. "
                "Auto-generating placeholder names."
            )
            args.class_names = [f"Class_{i}" for i in range(args.num_classes)]
            if args.num_classes > 0:
                args.class_names[0] = "Background"

    model = get_model(num_classes=args.num_classes, pretrained_backbone=bool(args.pretrained_backbone))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    threshold = float(args.score_threshold)
    class_names = args.class_names

    if args.webcam:
        # Webcam is interactive, showing window is the default expectation.
        run_webcam_inference(
            model=model,
            device=device,
            webcam_index=int(args.webcam_index),
            webcam_output=str(args.webcam_output),
            class_names=class_names,
            threshold=threshold,
            show_plot=True if args.show else True,
            save_output=bool(args.save),
        )
        return

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
        raise ValueError("Provide at least one input: --image, --image-dir, --video, or --webcam")

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