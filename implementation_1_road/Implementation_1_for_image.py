import cv2
import numpy as np
import os


def load_image(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def add_search_window(image, ratio=(0.5, 0.4)):  # 50% width, 40% height, positioned centrally
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
        cv2.imwrite("1_search_window.jpg", search_window_img)

        return search_window_img, (top_left_x, top_left_y, window_width, window_height)


def build_pyramid_in_roi(gray, roi_coords):
    top_left_x, top_left_y, width, height = roi_coords
    # Crop search window
    roi = gray[top_left_y:top_left_y + height, top_left_x:top_left_x + width]
    cv2.imwrite("2_search_window_cropped.jpg", roi)

    level1 = cv2.pyrDown(roi)  # Subsampling by 2x
    level2 = cv2.pyrDown(level1)

    cv2.imwrite("3_pyramid_level2.jpg", level2)
    return roi, level2


def adjust_exposure_contrast(image):
    alpha = 2.0  # Contrast control (1.0-3.0)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha)
    cv2.imwrite("4_adjusted_image.jpg", adjusted)
    return adjusted


def compute_gradient(partial_image):
    # Apply Gaussian Blur to reduce noise
    # blurred_image = cv2.GaussianBlur(partial_image, (3, 3), 1)
    # cv2.imwrite("5_after_blur.jpg", blurred_image)

    # Apply Canny edge detection
    edges = cv2.Canny(partial_image, threshold1=100, threshold2=200)
    cv2.imwrite("6_edges.jpg", edges)

    return edges


def detect_road_edges_from_bottom(image):
    """
    Detect road edges by considering the bottom-left and bottom-right points.
    """
    # Ensure the image is in grayscale (single channel)
    if len(image.shape) == 3:  # If the image is in BGR (3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize the bottom-left and bottom-right edge variables
    bottom_left = None
    bottom_right = None


    # Iterate over contours to find bottom-left and bottom-right road edges
    for contour in contours:
        for point in contour:
            x, y = point[0]
            if bottom_left is None or (y > bottom_left[1] and x < bottom_left[0]):
                bottom_left = (x, y)
            if bottom_right is None or (y > bottom_right[1] and x > bottom_right[0]):
                bottom_right = (x, y)

    # Debugging: Print the detected points
    print(f"Bottom-left point: {bottom_left}, Bottom-right point: {bottom_right}")

    # Convert back to color (BGR) to draw red points
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Mark bottom-left and bottom-right edges with red points
    if bottom_left:
        cv2.circle(image_color, bottom_left, 5, (0, 0, 255), -1)
    if bottom_right:
        cv2.circle(image_color, bottom_right, 5, (0, 0, 255), -1)

    cv2.imwrite("7_road_edges_from_bottom_debug.jpg", image_color)
    return image_color, bottom_left, bottom_right


def draw_vertical_lines(image, bottom_left, bottom_right):
    """
    Optional, only to visualize if road edges are detected correct
    Draws two vertical lines at the positions of the bottom-left and bottom-right points, spanning the full height.
    """
    if bottom_left and bottom_right:
        # Get the x-coordinates of the bottom-left and bottom-right points
        x_bottom_left = bottom_left[0]
        x_bottom_right = bottom_right[0]

        # Get the image height
        height = image.shape[0]

        # Draw the vertical lines
        cv2.line(image, (x_bottom_left, 0), (x_bottom_left, height), (0, 0, 255), 2)  # Line at bottom-left
        cv2.line(image, (x_bottom_right, 0), (x_bottom_right, height), (0, 0, 255), 2)  # Line at bottom-right

    cv2.imwrite("8_lines.jpg", image)
    return image


def crop_between_lines(image, bottom_left, bottom_right):
    """
    Crops the image between the bottom-left and bottom-right vertical lines.
    """
    if bottom_left and bottom_right:
        # Get the x-coordinates of the bottom-left and bottom-right points
        x_bottom_left = bottom_left[0]
        x_bottom_right = bottom_right[0]

        # Crop the image between these x-coordinates
        cropped_image = image[:, x_bottom_left:x_bottom_right]

        return cropped_image
    else:
        return image  # If no coord found, return the original image


def perform_detection_on_roi(roi, net, layer_names, confidence_threshold=0.5):
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
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    with open(class_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, layer_names, classes


# Main Execution
image_path = "frame1.png"

cfg_path = "files/yolov4-tiny.cfg"  # YOLO Lite cfg path
weights_path = "files/yolov4-tiny.weights"  # YOLO Lite weights path
class_file = "files/coco.names"  # Class file for COCO classes

# Load YOLO Lite model
net, layer_names, classes = load_yolo_lite_model(cfg_path, weights_path, class_file)

# Load image
img, gray = load_image(image_path)

# Add search window to image and get ROI coordinates
img_with_search_window, roi_coords = add_search_window(gray)

# Build pyramid for ROI
roi, level2 = build_pyramid_in_roi(gray, roi_coords)

# Adjust exposure and contrast
adjusted_image = adjust_exposure_contrast(level2)

gradient_image = compute_gradient(adjusted_image)

# Detect road edges from bottom-left and bottom-right
output_image, bottom_left, bottom_right = detect_road_edges_from_bottom(gradient_image)

# Draw the parallel lines on the image
# output_image_with_lines = draw_vertical_lines(output_image, bottom_left, bottom_right)

# Crop the image between the vertical lines
updated_roi = crop_between_lines(level2, bottom_left, bottom_right)
cv2.imwrite("final_updated_roi.jpg", updated_roi)

# Detect objects in the ROI (optional)
boxes, confidences, class_ids = perform_detection_on_roi(updated_roi, net, layer_names)

# Draw bounding boxes and labels on the ROI
for i in range(len(boxes)):
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]

    # Draw rectangle and label
    cv2.rectangle(updated_roi, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(updated_roi, f"{label}", (x - 15, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.putText(updated_roi, f"{confidence:.2f}", (x - 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

# Display the image with detected objects in the updated ROI
cv2.imshow("Detected Objects in Updated ROI", updated_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the result
cv2.imwrite(os.path.join("detected_updated_roi.jpg"), updated_roi)
