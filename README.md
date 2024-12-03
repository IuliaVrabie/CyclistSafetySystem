# CyclistSafetySystem

## Overview

The **Cyclist Safety System** aims to enhance cyclist safety by detecting obstacles, such as pedestrians, vehicles, and other bikes, in real-time. This system is designed to be implemented on a Raspberry Pi 4, camera and a laser range finder to help detect hazards from a distance. The system activates an alarm when a potential collision is detected.

## Features

- **Object Detection**: Detects pedestrians, bicycles, and vehicles using YOLO Lite or MobileNet SSD models.
- **Search Window**: Automatically identifies a region of interest (ROI) in each video frame for efficient hazard detection.
- **Pyramid Image Processing**: Uses image pyramids to improve detection accuracy at different scales.
- **Real-time Video Processing**: Analyzes video frames for object detection, drawing bounding boxes around detected objects.
- **Automatic Frame Saving**: Saves frames with detected objects for later review.
- **Customizable Detection Models**: Supports YOLO Lite and MobileNet SSD models for object detection.

## Code Explanation

### Functions:

1. **add_search_window**: Adds a search window to the image, highlighting a central region for object detection.
2. **build_pyramid_in_roi**: Creates an image pyramid for the region of interest (ROI) to improve multi-scale detection.
3. **apply_roi_mask**: Applies a triangular mask to image to detect only in this triangular region.
5. **perform_SSD_detection_on_roi**: Detects objects in the ROI using the MobileNet SSD model and returns bounding boxes.
7. **load_mobilenet_ssd_model**: Loads the MobileNet SSD model for object detection.
8. **draw_bounding_boxes**: Draws bounding boxes around detected objects with labels and confidence scores.

### Workflow:

- **Frame Processing**: Each video frame is converted to grayscale, and a search window is applied to focus on potential hazard areas.
- **Object Detection**: The system uses MobileNet SSD to detect objects in the ROI and draws bounding boxes around them.
- **Frame Saving**: Frames with detected objects are saved for later review.
- **Real-time Display**: Processed frames are shown in real-time, with the option to stop playback by pressing 'q'.
  
### Model:

- **MobileNet SSD**: A pre-trained deep learning model designed for object detection tasks, including detecting people, cars, and bicycles.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/IuliaVrabie/CyclistSafetySystem.git
   ```
2. Install the required dependencies:
   ```bash
   pip install opencv-python numpy tensorflow
   ```
3. Download the necessary model files for MobileNet SSD and place them in the `files/` directory.
  - https://github.com/MediosZ/MobileNet-SSD/blob/master/mobilenet/MobileNetSSD_deploy.prototxt
  - https://github.com/PINTO0309/MobileNet-SSD-RealSense/blob/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel
4. Adjust the paths in the code to point to your video file and model weights.

## Usage

Run the system with:
```bash
python Model_Detection_Video.py
```

The system will process the video and detect objects, saving frames with detections to the `detected_frames` folder.

## Requirements

- Raspberry Pi 4 Model B
- Camera
- Laser range finder
- Python 3.x
- OpenCV
- Pre-trained YOLO Lite or MobileNet SSD models

## Contributing

## License
