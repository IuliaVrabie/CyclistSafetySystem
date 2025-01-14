import time
import cv2
import numpy as np
import os

def add_search_window(image, ratio=(0.2, 0.3)):  # 50% width, 40% height, positioned centrally
    if len(image.shape) == 2:
        height, width = image.shape

        window_width = int(width * ratio[0])
        window_height = int(height * ratio[1])
        top_left_x = (width - window_width) // 2
        top_left_y = (height - window_height) // 2 + 50  # Shift down by 50px
        search_window_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR to draw colored rectangle

        cv2.rectangle(
            search_window_img,
            (top_left_x, top_left_y),
            (top_left_x + window_width, top_left_y + window_height),
            (0, 255, 0),  # Green color
            thickness=2
        )

        return search_window_img, (top_left_x, top_left_y, window_width, window_height)


def build_pyramid_in_roi(gray, roi_coords):
    top_left_x, top_left_y, width, height = roi_coords

    # Crop search window
    cropped_search_window = gray[top_left_y:top_left_y + height, top_left_x:top_left_x + width]

    level1 = cv2.pyrDown(cropped_search_window)  # Subsampling by 2x

    return cropped_search_window, level1


def apply_roi_mask(original_img_path, output_img_path, bbox):
    """
    Apply a triangular ROI mask on the detected region (person) in the image.

    Args:
    - original_img_path: str, path to the original image
    - output_img_path: str, path where the resulting image will be saved
    - bbox: tuple, bounding box (x, y, w, h) from YOLO detection

    Returns:
    - None (saves the output image to the given path)
    """

    # Load the original image
    original_img = cv2.imread(original_img_path)

    # Get the coordinates of the bounding box
    x, y, w, h = bbox

    # Create a mask for the triangular region
    mask = np.zeros_like(original_img)
    height, width = original_img.shape[:2]

    # Create the triangle points for the mask
    pts = np.array([[0, 0], [width, 0], [width // 2, height]], dtype=np.int32)

    # Fill the mask with the triangular shape
    cv2.fillPoly(mask, [pts], (255, 255, 255))

    # Apply the mask to the image, focusing on the region within the bounding box
    roi = original_img[y:y + h, x:x + w]
    roi_masked = cv2.bitwise_and(roi, roi, mask=cv2.cvtColor(mask[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY))

    # Place the masked ROI back into the original image
    original_img[y:y + h, x:x + w] = roi_masked

    # Save the result to the specified output path
    cv2.imwrite(output_img_path, original_img)


def perform_YOLO_detection_on_roi(roi, net, layer_names, confidence_threshold=0.5):
    # Convert grayscale ROI to RGB (3 channels) if necessary
    if len(roi.shape) == 2:  # If the ROI is grayscale (1 channel)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel image

    # Convert the ROI to a blob suitable for YOLO input
    blob = cv2.dnn.blobFromImage(roi, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform forward pass to get output layers
    outputs = net.forward(layer_names)

    boxes, confidences, class_ids = [], [], []
    height, width = roi.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Get rectangle coordinates for bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


def load_yolo_lite_model(cfg_path, weights_path, class_file):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Correct layer indexing

    with open(class_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, layer_names, classes


def draw_bounding_boxes(frame, boxes, confidences, class_ids, classes, target_classes):
    for i, box in enumerate(boxes):
        if class_ids[i] in target_classes:
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


video_path = '../amster1min.mp4'

# Yolo Model
cfg_path = "files/yolov4-tiny.cfg"  # YOLO Lite cfg
weights_path = "files/yolov4-tiny.weights"  # YOLO Lite weights
class_file = "files/coco.names"  # Class file for COCO classes

# Load YOLO Lite model
yolo_net, layer_names, classes = load_yolo_lite_model(cfg_path, weights_path, class_file)

# Define target classes (Person, Car, Bicycle)
yolo_target_classes = [0, 1, 2]  # Person, Bicycle, Car

# Create a directory to save frames where something is detected
output_dir = 'detected_frames'
os.makedirs(output_dir, exist_ok=True)

# Initialize the video capture object
cap = cv2.VideoCapture(video_path)

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get the frame rate (frames per second)
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Calculate the total video duration in seconds
video_duration = total_frames / frame_rate

# Loop through all frames to process the video
start_time = time.time()

frame_count = 0  # Frame counter to save frames with detections

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame or end of video.")
        break

    # Time for video playback
    original_video_time = time.time() - start_time

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Add search window to image and get ROI coordinates
    img_with_search_window, roi_coords = add_search_window(gray_frame)

    # Build pyramid for ROI
    cropped_search_window, pyramid_frame = build_pyramid_in_roi(gray_frame, roi_coords)


    # YoloLite

    # Perform detection on ROI using YOLO Lite
    boxes, confidences, class_ids = perform_YOLO_detection_on_roi(pyramid_frame, yolo_net, layer_names)

    # Draw bounding boxes around detected objects in the frame
    frame_with_boxes = draw_bounding_boxes(pyramid_frame, boxes, confidences, class_ids, classes, yolo_target_classes)



    # If detections exist, save the frame
    if len(boxes) > 0:
        frame_count += 1
        save_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(save_path, frame_with_boxes)


    # Display the frame with bounding boxes
    cv2.imshow('Video Playback', frame_with_boxes)

    # Press 'q' to quit the video playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Calculate total time for both original video playback and preprocessing
video_playback_time = time.time() - start_time  # total time for the entire video playback
speed = video_playback_time / video_duration

print(f'Original {video_duration}, Running {video_playback_time}, x{speed}')
