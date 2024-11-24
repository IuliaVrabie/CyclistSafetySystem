import time
import cv2
import numpy as np
import os
from collections import defaultdict


# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection
    x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    intersection_area = x_intersection * y_intersection
    area1 = w1 * h1
    area2 = w2 * h2

    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


# Function to calculate Precision and Recall
def calculate_precision_recall(true_boxes, pred_boxes, iou_threshold=0.5):
    """
    Calculate Precision and Recall based on IoU threshold.
    """
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    # For each predicted box, check if it has a corresponding ground truth box
    for pred_box in pred_boxes:
        max_iou = 0
        for true_box in true_boxes:
            iou = calculate_iou(pred_box, true_box)
            max_iou = max(max_iou, iou)
        if max_iou >= iou_threshold:
            tp += 1
        else:
            fp += 1

    # For ground truth boxes, check if they were matched with a predicted box
    for true_box in true_boxes:
        matched = False
        for pred_box in pred_boxes:
            iou = calculate_iou(pred_box, true_box)
            if iou >= iou_threshold:
                matched = True
                break
        if not matched:
            fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return precision, recall


# Function to compute mAP
def compute_mAP(true_boxes, pred_boxes, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    """
    Compute mean Average Precision (mAP) across different IoU thresholds.
    """
    precisions = []
    recalls = []

    for iou_threshold in iou_thresholds:
        precision, recall = calculate_precision_recall(true_boxes, pred_boxes, iou_threshold)
        precisions.append(precision)
        recalls.append(recall)

    mAP = np.mean(precisions)
    return mAP, precisions, recalls


# Function to compare models based on mAP, Precision, and Recall
def compare_models(true_boxes, yolo_pred_boxes, ssd_pred_boxes, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    """
    Compare YOLO Lite and MobileNet SSD models using mAP, Precision, and Recall.
    """
    print("Comparing YOLO Lite and MobileNet SSD models:")

    # Calculate mAP for YOLO Lite
    yolo_mAP, yolo_precisions, yolo_recalls = compute_mAP(true_boxes, yolo_pred_boxes, iou_thresholds)
    print(f"YOLO Lite mAP: {yolo_mAP:.4f}")
    print(f"YOLO Lite Precision: {np.mean(yolo_precisions):.4f}, Recall: {np.mean(yolo_recalls):.4f}")

    # Calculate mAP for MobileNet SSD
    ssd_mAP, ssd_precisions, ssd_recalls = compute_mAP(true_boxes, ssd_pred_boxes, iou_thresholds)
    print(f"MobileNet SSD mAP: {ssd_mAP:.4f}")
    print(f"MobileNet SSD Precision: {np.mean(ssd_precisions):.4f}, Recall: {np.mean(ssd_recalls):.4f}")

    return {
        "YOLO Lite": {"mAP": yolo_mAP, "Precision": np.mean(yolo_precisions), "Recall": np.mean(yolo_recalls)},
        "MobileNet SSD": {"mAP": ssd_mAP, "Precision": np.mean(ssd_precisions), "Recall": np.mean(ssd_recalls)}
    }


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


def perform_YOLO_detection_on_roi(roi, net, layer_names, confidence_threshold=0.5):
    if len(roi.shape) == 2:  # If the ROI is grayscale (1 channel)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel image

    blob = cv2.dnn.blobFromImage(roi, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

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

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


def perform_SSD_detection_on_roi(roi, net, confidence_threshold=0.5):
    if len(roi.shape) == 2:  # If the ROI is grayscale (1 channel)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel image

    blob = cv2.dnn.blobFromImage(roi, 1 / 255.0, (300, 300), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    detections = net.forward()

    boxes, confidences, class_ids = [], [], []
    height, width = roi.shape[:2]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
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


def load_mobilenet_ssd_model(prototxt_path, model_path):
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    class_labels = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
        'train', 'tvmonitor'
    ]

    return net, class_labels


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


# Example of running the comparison:
video_path = 'amster1min.mp4'
cfg_path = "files/yolov4-tiny.cfg"
weights_path = "files/yolov4-tiny.weights"
class_file = "files/coco.names"
prototxt_path = "files/MobileNetSSD_deploy.prototxt"
model_path = "files/MobileNetSSD_deploy.caffemodel"

# Load models
yolo_net, layer_names, classes = load_yolo_lite_model(cfg_path, weights_path, class_file)
ssd_net, class_labels = load_mobilenet_ssd_model(prototxt_path, model_path)

# Define target classes (Person, Car, Bicycle)
yolo_target_classes = [0, 1, 2]  # Person, Bicycle, Car
ssd_target_classes = [15, 7, 2]  # Person, Car, Bicycle

# Ground truth boxes for comparison (example)
true_boxes = [...]  # Replace with actual ground truth data
yolo_pred_boxes = [...]  # Replace with actual YOLO predictions
ssd_pred_boxes = [...]  # Replace with actual SSD predictions

# Run the comparison
results = compare_models(true_boxes, yolo_pred_boxes, ssd_pred_boxes)
