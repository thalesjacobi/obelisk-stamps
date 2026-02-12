# Stamp Detector ML Module

A reusable YOLO-based stamp detection system for identifying and extracting individual stamps from multi-stamp images.

## Features

- Detects individual stamps in images containing multiple stamps (2-up, 3-up, grids, etc.)
- Returns bounding boxes, confidence scores, and cropped images
- Can be used by any scraper in the project
- Falls back to legacy detection if model is not trained

## Quick Start

### 1. Install Dependencies

```bash
pip install ultralytics labelme
```

### 2. Prepare Training Data

```bash
# Sample 200 images from your scraped data
python ml/stamp_detector/prepare_data.py --sample 200

# Check status
python ml/stamp_detector/prepare_data.py --stats
```

### 3. Label Images with LabelMe

```bash
# Launch the labeling tool
python ml/stamp_detector/prepare_data.py --launch-labelme
```

In LabelMe:
1. Open an image
2. Press 'R' or click "Create Rectangle"
3. Draw a box around each stamp
4. Label it as "stamp"
5. Save (Ctrl+S)
6. Next image (D key)

**Label at least 50-100 images** for basic training, 200+ for better results.

### 4. Convert Annotations

```bash
python ml/stamp_detector/prepare_data.py --convert-labelme
```

### 5. Train the Model

```bash
# Basic training
python ml/stamp_detector/train.py

# With custom settings
python ml/stamp_detector/train.py --epochs 150 --batch 8 --model s
```

### 6. Use in Your Code

```python
from ml.stamp_detector import StampDetector

# Initialize detector
detector = StampDetector()

# Check if model is ready
if detector.is_ready:
    # Count stamps in an image
    count = detector.count_stamps("path/to/image.jpg")

    # Check if multi-stamp
    if detector.is_multi_stamp("path/to/image.jpg"):
        # Get cropped images
        crops = detector.detect_and_crop("path/to/image.jpg")

        # Or save to files
        paths = detector.detect_and_save("path/to/image.jpg")
```

## File Structure

```
ml/stamp_detector/
├── __init__.py          # Module exports
├── detector.py          # StampDetector class
├── train.py             # Training script
├── prepare_data.py      # Data preparation tools
├── config.yaml          # YOLO training config
├── README.md            # This file
├── weights/             # Trained model weights
│   └── stamp_detector.pt
└── training_data/       # Training images and labels
    ├── images/          # Images to label
    └── labels/          # YOLO format labels (.txt)
```

## YOLO Label Format

Each image needs a `.txt` file with the same name in the `labels/` folder.
Each line represents one stamp: `class_id center_x center_y width height`

Example (`image001.txt`):
```
0 0.25 0.5 0.4 0.8
0 0.75 0.5 0.4 0.8
```

- `class_id`: Always 0 (for "stamp")
- `center_x`, `center_y`: Center of bounding box (normalized 0-1)
- `width`, `height`: Box dimensions (normalized 0-1)

## Integration with Scrapers

The detector is integrated with `scrape_postbeeld.py`. To enable:

1. Train the model (steps 1-5 above)
2. Set in `scrape_postbeeld.py`:
   ```python
   ENABLE_IMAGE_SPLITTING = True
   USE_ML_DETECTOR = True
   ```

Other scrapers can import and use the detector the same way:

```python
from ml.stamp_detector import StampDetector

detector = StampDetector()
if detector.is_ready and detector.is_multi_stamp(image_path):
    parts = detector.detect_and_save(image_path)
```

## Tips for Labeling

1. **Be consistent**: Draw boxes tight around stamps, including perforations
2. **Include variety**: Label single stamps, 2-up, 3-up, and grid images
3. **Handle edge cases**: Include partially visible stamps, rotated stamps
4. **Label everything**: Even if a stamp is partially cut off, label it
5. **Watch for watermarks**: Don't include watermarks in the stamp boxes

## Troubleshooting

**"ML model not trained yet"**: Run the training steps above.

**"ultralytics not installed"**: Run `pip install ultralytics`

**Training is slow**: Reduce batch size (`--batch 4`) or use smaller model (`--model n`)

**Poor detection**: Label more images, especially images similar to those failing.
