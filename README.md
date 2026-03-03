# Faster R-CNN (Train + Inference)

This project trains and runs inference for Faster R-CNN (ResNet-50 FPN) on COCO-format datasets.

## Project Files

- `train.py`: Training script driven by `train.yaml`.
- `train.yaml`: Training configuration (dataset, model, optimizer, scheduler, epochs, outputs).
- `test.py`: Inference script for image, folder, video, and webcam.
- `requirements.txt`: Python dependencies.

## Setup

1. Create and activate your environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run training:

```bash
python train.py --config train.yaml
```

### What `train.py` does

- Loads COCO dataset paths from `train.yaml`.
- Reads `categories` from your COCO annotation file.
- Builds a contiguous label map for Faster R-CNN:
  - `cat_id_to_contig` (COCO category ID -> model class index)
  - `contig_to_cat_id` (model class index -> COCO category ID)
- Trains Faster R-CNN and saves checkpoints in `output.dir`.
- Each checkpoint stores:
  - `model_state`
  - `num_classes`
  - `class_names`
  - `cat_id_to_contig`
  - `contig_to_cat_id`

Saved files:

- `outputs/fasterrcnn_epoch_<N>.pth`
- `outputs/fasterrcnn_final.pth`

## Inference (`test.py`)

`test.py` does not read `train.yaml`.

### 1) Single image

```bash
python test.py --checkpoint outputs/fasterrcnn_final.pth --image test.jpg
```

### 2) Image folder

```bash
python test.py --checkpoint outputs/fasterrcnn_final.pth --image-dir path/to/images
```

### 3) Video file

```bash
python test.py --checkpoint outputs/fasterrcnn_final.pth --video input.mp4 --video-output outputs/predictions/output.mp4
```

### 4) Webcam

```bash
python test.py --checkpoint outputs/fasterrcnn_final.pth --webcam --webcam-index 0 --webcam-output outputs/predictions/webcam.mp4
```

Press `q` to quit webcam/video display window.

## Inference Arguments

Required:

- `--checkpoint PATH`

Input mode (choose one):

- `--image PATH`
- `--image-dir PATH`
- `--video PATH`
- `--webcam`

Common optional:

- `--device auto|cpu|cuda`
- `--score-threshold 0.5`
- `--show`
- `--save` / `--no-save`

Image mode optional:

- `--output-dir outputs/predictions`
- `--figsize 12 10`

Video mode optional:

- `--video-output outputs/predictions/pred_video.mp4`

Webcam mode optional:

- `--webcam-index 0`
- `--webcam-output outputs/predictions/pred_webcam.mp4`

Advanced optional:

- `--num-classes N`
- `--class-names name0 name1 ...`
- `--pretrained-backbone`

Note: `test.py` loads metadata from checkpoint when available and may override `--num-classes` / `--class-names` if checkpoint values are more accurate.
