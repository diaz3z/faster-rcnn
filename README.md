# Faster R-CNN Training and Inference

This project trains and tests a Faster R-CNN (ResNet-50 FPN) model using COCO-format annotations.

## Files

- `train.py`: Trains the model using config values from `train.yaml`.
- `test.py`: Runs inference on image(s) or video using only command-line arguments.
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

### Image

```bash
python test.py --checkpoint outputs/fasterrcnn_final.pth --image test.jpg
```

### Image directory

```bash
python test.py --checkpoint outputs/fasterrcnn_final.pth --image-dir path/to/images
```

### Video

```bash
python test.py --checkpoint outputs/fasterrcnn_final.pth --video input.mp4 --video-output outputs/predictions/output.mp4
```

### Common optional args

- `--num-classes 4`
- `--class-names Background Chair Person Table`
- `--device auto`
- `--score-threshold 0.5`
- `--show`
- `--no-save`

For image mode only:

- `--output-dir outputs/predictions`
- `--figsize 12 10`

## Notes

- `test.py` does not read `train.yaml`.
- Video inference requires OpenCV.
