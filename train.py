import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F


class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_coco_dataset(img_dir: str, ann_file: str) -> CocoDetection:
    return CocoDetection(root=img_dir, annFile=ann_file, transforms=CocoTransform())


def collate_fn(batch):
    return tuple(zip(*batch))


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


def build_targets(targets: List[Any], device: torch.device):
    processed_targets = []
    valid_indices = []

    for i, target in enumerate(targets):
        boxes = []
        labels = []

        for obj in target:
            bbox = obj.get("bbox")
            category_id = obj.get("category_id")
            if bbox is None or category_id is None or len(bbox) != 4:
                continue

            x, y, w, h = bbox
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(int(category_id))

        if boxes:
            processed_targets.append(
                {
                    "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
                    "labels": torch.tensor(labels, dtype=torch.int64, device=device),
                }
            )
            valid_indices.append(i)

    return processed_targets, valid_indices


def train_one_epoch(model, optimizer, data_loader, device, epoch: int) -> float:
    model.train()
    running_loss = 0.0
    steps = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        processed_targets, valid_indices = build_targets(list(targets), device)

        if not processed_targets:
            continue

        valid_images = [images[idx] for idx in valid_indices]
        loss_dict = model(valid_images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        steps += 1

    avg_loss = running_loss / steps if steps > 0 else 0.0
    print(f"Epoch [{epoch}] avg_loss={avg_loss:.4f} steps={steps}")
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN from YAML config")
    parser.add_argument("--config", type=str, default="train.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_cfg = str(cfg.get("device", "auto")).lower()
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)

    train_cfg = cfg["dataset"]["train"]
    val_cfg = cfg["dataset"].get("val")

    train_dataset = get_coco_dataset(train_cfg["img_dir"], train_cfg["ann_file"])
    val_dataset = None
    if val_cfg and val_cfg.get("img_dir") and val_cfg.get("ann_file"):
        val_dataset = get_coco_dataset(val_cfg["img_dir"], val_cfg["ann_file"])

    dl_cfg = cfg.get("dataloader", {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(dl_cfg.get("batch_size", 4)),
        shuffle=bool(dl_cfg.get("shuffle", True)),
        num_workers=int(dl_cfg.get("num_workers", 0)),
        collate_fn=collate_fn,
    )

    if val_dataset is not None:
        _ = DataLoader(
            val_dataset,
            batch_size=int(dl_cfg.get("batch_size", 4)),
            shuffle=False,
            num_workers=int(dl_cfg.get("num_workers", 0)),
            collate_fn=collate_fn,
        )

    model_cfg = cfg["model"]
    model = get_model(
        num_classes=int(model_cfg["num_classes"]),
        pretrained=bool(model_cfg.get("pretrained", True)),
    )
    model.to(device)

    opt_cfg = cfg["optimizer"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = str(opt_cfg.get("name", "SGD")).lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=float(opt_cfg.get("lr", 1e-4)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        )
    else:
        optimizer = torch.optim.SGD(
            params,
            lr=float(opt_cfg.get("lr", 0.005)),
            momentum=float(opt_cfg.get("momentum", 0.9)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0005)),
        )

    scheduler = None
    sch_cfg = cfg.get("scheduler", {})
    if bool(sch_cfg.get("enabled", True)):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch_cfg.get("step_size", 3)),
            gamma=float(sch_cfg.get("gamma", 0.1)),
        )

    out_cfg = cfg.get("output", {})
    output_dir = Path(out_cfg.get("dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(out_cfg.get("save_every", 1))
    ckpt_pattern = out_cfg.get("checkpoint_pattern", "fasterrcnn_resnet50_epoch_{epoch}.pth")
    final_model_name = out_cfg.get("final_model_name", "fasterrcnn_final.pth")

    num_epochs = int(cfg.get("train", {}).get("num_epochs", 5))

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch)

        if scheduler is not None:
            scheduler.step()

        if save_every > 0 and epoch % save_every == 0:
            ckpt_name = ckpt_pattern.format(epoch=epoch)
            ckpt_path = output_dir / ckpt_name
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = output_dir / final_model_name
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
