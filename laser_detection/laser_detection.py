import numpy as np
import cv2

# Constants
LIGHT_SPEED = 3.0e8  # m/s
FOCAL_LENGTH = 800
IMAGE_HEIGHT = 1000
IMAGE_WIDTH = 1500
LX = 0  # Horizontal distance between laser and webcam
LY = 1  # Vertical distance between laser and webcam
LZ = 2  # Height difference between laser and webcam
TF_MINI_S_MAX_RANGE = 12  # Maximum range in meters
HAAR_CLASSIFIER_PEDESTRIAN = "haarcascade_fullbody.xml"
HAAR_CLASSIFIER_VEHICLE = "cars.xml"

# Initialize Haar cascades
pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAAR_CLASSIFIER_PEDESTRIAN)
vehicle_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + HAAR_CLASSIFIER_VEHICLE)


def calculate_distance(time_of_flight):
    """Calculate distance using time-of-flight."""
    return (LIGHT_SPEED * time_of_flight) / 2


def calculate_image_coordinates(x, y):
    """Translate laser coordinates to image coordinates."""
    u = (FOCAL_LENGTH * LZ / (y + LY)) + (IMAGE_HEIGHT / 2)
    v = (FOCAL_LENGTH * x / (y + LY)) + (IMAGE_WIDTH / 2)
    return int(u), int(v)


def detect_breakpoints(distance_data):
    """Detect breakpoints in laser data."""
    diff = np.diff(distance_data)
    breakpoints = np.where(np.abs(diff) > 0.2)[0]  # Threshold: significant change
    return breakpoints


def classify_objects(image, roi):
    """Classify objects using Haar cascades."""
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pedestrians = pedestrian_cascade.detectMultiScale(gray_roi, 1.1, 5)
    vehicles = vehicle_cascade.detectMultiScale(gray_roi, 1.1, 5)

    if len(pedestrians) > 0:
        return "Pedestrian"
    elif len(vehicles) > 0:
        return "Vehicle"
    else:
        return "Unknown"


def process_frame(frame, laser_data):
    """Process a frame from the webcam."""
    for x, y, dist in laser_data:
        # Convert laser data to image coordinates
        u, v = calculate_image_coordinates(x, y)

        # Define ROI around the detected laser point
        roi_size = 100
        roi = frame[max(0, u - roi_size):u + roi_size, max(0, v - roi_size):v + roi_size]

        if roi.size > 0:
            object_type = classify_objects(frame, roi)
            cv2.putText(frame, object_type, (v, u - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.rectangle(frame, (v - roi_size, u - roi_size), (v + roi_size, u + roi_size), (0, 255, 0), 2)

    return frame


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Simulated laser data
        laser_data = [
            (3, 5, 2),  # (x, y, distance) sample
            (4, 6, 2.5),
        ]

        # Detect breakpoints in laser data
        distance_data = [d[2] for d in laser_data]
        breakpoints = detect_breakpoints(distance_data)

        # Process frame with laser data
        processed_frame = process_frame(frame, laser_data)

        # Display the result
        cv2.imshow("Processed Frame", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
