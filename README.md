# image Photo Classifier

This repository is a small PyTorch project that fine-tunes MobileNetV2 to classify image photos as "good" or "bad". It includes training and prediction scripts.

### 1. Setup
```bash
pip install -r requirements.txt
```

## Prepare Data
Place your images in:
- `data/bad/` - Bad image photos
- `data/good/` - Good image photos

Aim for at least 100-200 images per class for decent results.

## Train
```bash
python train.py
```

## Predict
```bash
python predict.py path/to/photo.jpg
```

## Notes
- Uses MobileNetV2 with transfer learning (efficient for 12GB GPU)
- Mixed precision training (FP16) for memory efficiency
- 80/20 train/validation split
- Trains for 10 epochs (adjust in train.py if needed)
