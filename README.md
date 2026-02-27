# Faster R-CNN Training and Inference

This project trains and tests a Faster R-CNN (ResNet-50 FPN) model using COCO-format annotations.

## Files

- `train.py`: Trains the model using config values from `train.yaml`.
- `test.py`: Runs inference on image(s) using command-line arguments.
- `train.yaml`: Central config for training dataset paths, model setup, optimizer/scheduler, and output settings.
- `requirements.txt`: Python dependencies.

## Setup

1. Create and activate a Python environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Format

Expected COCO-style layout (default in `train.yaml`):

```text
Dataset/
  train/
    annotations/
      annotations.json
    *.jpg / *.png ...
  val/
    annotations/
      annotations.json
    *.jpg / *.png ...
```

Update paths in `train.yaml` if your layout differs.

## Train

```bash
python train.py --config train.yaml
```

By default, checkpoints are saved to `outputs/`.

## Test (Inference)

Single image:

```bash
python test.py --config train.yaml --checkpoint outputs/fasterrcnn_final.pth --image test.jpg
```

Run on a directory:

```bash
python test.py --config train.yaml --checkpoint outputs/fasterrcnn_final.pth --image-dir path/to/images
```

Optional test arguments:

- `--output-dir outputs/predictions`
- `--score-threshold 0.5`
- `--figsize 12 10`
- `--show`
- `--no-save`

## Configuration

Edit `train.yaml` to change:

- Dataset paths
- Number of classes and class names
- Batch size and workers
- Optimizer/scheduler settings
- Epoch count
- Output/checkpoint naming
