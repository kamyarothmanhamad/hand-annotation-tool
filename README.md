# COCO Dataset Manual Annotation Tool

This is a simple tool for manually annotating images from the COCO dataset. The tool allows you to draw keypoints, curves, and bounding boxes on images, and save the annotations in a compatible format.

## Features

- Load and browse through images from a COCO dataset directory
- Draw keypoints (e.g., joints of a person)
- Draw curves (for segmentation masks)
- Draw bounding boxes (for object detection)
- Save annotations in JSON format
- Load existing annotations for further editing

## Requirements

- Python 3.6+
- Pillow
- NumPy
- Tkinter (usually comes with Python)

Install the required packages:

```
pip install -r requirements.txt
```

## Usage

1. Run the application:

```
python coco_annotator.py
```

2. Click "Load Dataset" and select the directory containing your COCO dataset images.
3. Use the radio buttons to select the annotation mode (Keypoint, Curve, or Bounding Box).
4. Draw annotations on the image:
   - **Keypoint**: Click to place a keypoint
   - **Curve**: Click to start a curve, click to add points, and close the curve by clicking near the starting point
   - **Bounding Box**: Click and drag to draw a bounding box
5. Use the tabs at the bottom to view and manage your annotations.
6. Click "Save Annotations" to save the annotations for the current image.
7. Use the "Previous" and "Next" buttons to navigate through images.

## Annotation Format

Annotations are saved in a JSON file with the following structure:

```json
{
  "image": "image_name.jpg",
  "keypoints": [
    {"id": 1, "x": 100, "y": 200},
    {"id": 2, "x": 150, "y": 250}
  ],
  "curves": [
    {"id": 1, "points": [{"x": 10, "y": 20}, {"x": 30, "y": 40}]}
  ],
  "bboxes": [
    {"id": 1, "x1": 50, "y1": 60, "x2": 150, "y2": 200}
  ]
}
```

## License


