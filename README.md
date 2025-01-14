# Cyclist Safety System

## Overview
The Cyclist Safety System is designed to enhance cyclist safety by detecting potential hazards in real time. The project incorporates hardware components like cameras and sensors, along with software implementations for hazard detection.

In this repo you can find two implementations, each designed to address safety in distinct ways:

1. **Camera-based Hazard Detection**
2. **LIDAR-based Hazard Detection**

---

## 1. Camera-based Hazard Detection

This implementation uses a camera to monitor the cyclist's surroundings and detect potential hazards, such as vehicles, pedestrians, and other cyclists. Machine learning algorithms are employed to process the video feed and identify obstacles in real time.


### Features

- **Object Detection**: Detects pedestrians, bicycles, and vehicles using YOLO model.
- **Search Window**: Automatically identifies a region of interest (ROI) for efficient hazard detection.
- **Pyramid Image Processing**: Uses image pyramids to reduced computational complexity.
- **Real-time Video Processing**: Analyzes video frames for object detection.
- **Automatic Frame Saving**: Saves frames with detected objects for later review.

### Requirements
- A Raspberry Pi 4 Model B (or equivalent).
- Python 3.7+.
- Pre-trained object detection model (e.g., YOLO).

### Installation
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

### Running the Code
```bash
python implementation_1_road/Implementation_1_for_image.py
```
or 
```bash
python implementation_1_road/Implementation_1_for_video.py
```

---

## 2. LIDAR-based Hazard Detection

This implementation employs a 360-degree LIDAR sensor to detect obstacles within a 30-meter range. The system processes distance data from the LIDAR to identify objects and determine their proximity.

### Features
- **360-degree Detection**: Covers the cyclist’s entire surroundings for comprehensive safety.
- **Distance Alerts**: Triggers alarms based on object proximity and the cyclist’s speed.
- **Brake Integration**: Optional feature to apply slight braking upon hazard detection.

### Requirements
- USB WebCamera
- RPLIDAR A1/A2 360 laser scanner.
- Python 3.7+.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/IuliaVrabie/CyclistSafetySystem.git
   ```
2. Navigate to the LIDAR-based implementation directory:
   ```bash
   cd CyclistSafetySystem/Implementation_2
   ``` 
3. Install dependencies:
   ```bash
   pip install opencv-python numpy tensorflow rplidar ultralytics
   ```
4. Connect and configure the LIDAR sensor manual.

### Running the Code
Run the LIDAR-based detection system using the following command:
```bash
python implementation_2_lidar.py
```
Ensure the LIDAR sensor is connected and functioning correctly.

---


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact
For further information or support, feel free to contact the project maintainer, [Iulia Vrabie](https://github.com/IuliaVrabie).

