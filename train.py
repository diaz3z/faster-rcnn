import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from tqdm import tqdm


class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes: int, pretrained: bool = True):
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


def read_coco_categories(ann_file: str) -> List[Dict[str, Any]]:
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = data.get("categories", [])
    if not cats:
        raise ValueError(f"No categories found in {ann_file}")
    return cats


def build_label_maps_from_coco(ann_file: str):
    cats = read_coco_categories(ann_file)
    cats_sorted = sorted(cats, key=lambda c: int(c["id"]))

    class_names = ["__background__"] + [str(c["name"]) for c in cats_sorted]

    cat_id_to_contig = {}
    contig_to_cat_id = {}

    for idx, c in enumerate(cats_sorted, start=1):
        cat_id = int(c["id"])
        cat_id_to_contig[cat_id] = idx
        contig_to_cat_id[idx] = cat_id

    return class_names, cat_id_to_contig, contig_to_cat_id


def resolve_class_names_and_maps(ann_file, cfg_class_names, cfg_num_classes):
    coco_class_names, cat_id_to_contig, contig_to_cat_id = build_label_maps_from_coco(
        ann_file
    )

    class_names = cfg_class_names if cfg_class_names else coco_class_names
    num_classes = len(class_names)

    if cfg_num_classes is not None and int(cfg_num_classes) != num_classes:
        raise ValueError(
            f"num_classes mismatch cfg={cfg_num_classes} actual={num_classes}"
        )

    return class_names, cat_id_to_contig, contig_to_cat_id, num_classes


def get_coco_dataset(img_dir: str, ann_file: str) -> CocoDetection:
    return CocoDetection(root=img_dir, annFile=ann_file, transforms=CocoTransform())


def build_targets(targets, device, cat_id_to_contig):
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
            if w <= 0 or h <= 0:
                continue

            cat_id = int(category_id)
            contig = cat_id_to_contig.get(cat_id)
            if contig is None:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(contig)

        if boxes:
            processed_targets.append(
                {
                    "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
                    "labels": torch.tensor(labels, dtype=torch.int64, device=device),
                }
            )
            valid_indices.append(i)

    return processed_targets, valid_indices


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    cat_id_to_contig,
):
    model.train()
    running_loss = 0.0
    steps = 0

    progress_bar = tqdm(
        data_loader,
        desc=f"Epoch {epoch}",
        leave=True,
    )

    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        processed_targets, valid_indices = build_targets(
            list(targets), device, cat_id_to_contig
        )

        if not processed_targets:
            continue

        valid_images = [images[idx] for idx in valid_indices]
        loss_dict = model(valid_images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value = float(losses.item())
        running_loss += loss_value
        steps += 1

        avg_loss = running_loss / steps
        progress_bar.set_postfix(
            loss=f"{loss_value:.4f}",
            avg=f"{avg_loss:.4f}",
        )

    avg_loss = running_loss / steps if steps > 0 else 0.0
    return avg_loss


def save_checkpoint(
    path,
    model,
    num_classes,
    class_names,
    cat_id_to_contig,
    contig_to_cat_id,
):
    ckpt = {
        "model_state": model.state_dict(),
        "num_classes": num_classes,
        "class_names": class_names,
        "cat_id_to_contig": cat_id_to_contig,
        "contig_to_cat_id": contig_to_cat_id,
    }
    torch.save(ckpt, path)
    print(f"Saved checkpoint {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.yaml")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_cfg = str(cfg.get("device", "auto")).lower()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_cfg == "auto"
        else torch.device(device_cfg)
    )

    train_cfg = cfg["dataset"]["train"]
    val_cfg = cfg["dataset"].get("val")

    train_dataset = get_coco_dataset(train_cfg["img_dir"], train_cfg["ann_file"])
    val_dataset = None
    if val_cfg and val_cfg.get("img_dir") and val_cfg.get("ann_file"):
        val_dataset = get_coco_dataset(val_cfg["img_dir"], val_cfg["ann_file"])

    model_cfg = cfg["model"]

    class_names, cat_id_to_contig, contig_to_cat_id, num_classes = (
        resolve_class_names_and_maps(
            ann_file=train_cfg["ann_file"],
            cfg_class_names=model_cfg.get("class_names"),
            cfg_num_classes=model_cfg.get("num_classes"),
        )
    )

    model = get_model(num_classes=num_classes, pretrained=model_cfg.get("pretrained", True))
    model.class_names = class_names
    model.to(device)

    dl_cfg = cfg.get("dataloader", {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(dl_cfg.get("batch_size", 4)),
        shuffle=bool(dl_cfg.get("shuffle", True)),
        num_workers=int(dl_cfg.get("num_workers", 0)),
        collate_fn=collate_fn,
    )

    opt_cfg = cfg["optimizer"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=float(opt_cfg.get("lr", 0.005)),
        momentum=float(opt_cfg.get("momentum", 0.9)),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0005)),
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg.get("scheduler", {}).get("step_size", 3)),
        gamma=float(cfg.get("scheduler", {}).get("gamma", 0.1)),
    )

    output_dir = Path(cfg.get("output", {}).get("dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = int(cfg.get("train", {}).get("num_epochs", 5))

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            cat_id_to_contig,
        )

        scheduler.step()

        print(f"Epoch {epoch} completed avg_loss={avg_loss:.4f}")

        save_checkpoint(
            output_dir / f"fasterrcnn_epoch_{epoch}.pth",
            model,
            num_classes,
            class_names,
            cat_id_to_contig,
            contig_to_cat_id,
        )

    save_checkpoint(
        output_dir / "fasterrcnn_final.pth",
        model,
        num_classes,
        class_names,
        cat_id_to_contig,
        contig_to_cat_id,
    )


if __name__ == "__main__":
    main()