import cv2
from rplidar import RPLidar
import threading
import time
import numpy as np
from ultralytics import YOLO

# LIDAR setup
PORT_NAME = 'COM3'

# Camera setup
FOCAL_LENGTH = 605
IMAGE_WIDTH = 640  # Camera resolution width
IMAGE_HEIGHT = 480  # Camera resolution height

# LIDAR to Camera offsets
LIDAR_TO_CAMERA_X = -0.05  # Horizontal offset
LIDAR_TO_CAMERA_Y = 2.5  # Adjusted for vertical alignment
LIDAR_TO_CAMERA_Z = 0.50  # Depth offset

# Rotation Angle
ROTATION_ANGLE = -85

# Global variable to store LIDAR data
lidar_data = []

# YOLOv8 model
model = YOLO('yolov8n.pt')


def initialize_lidar(port=PORT_NAME):
    """Initialize the RPLidar."""
    lidar = RPLidar(port)
    return lidar


def lidar_thread(lidar):
    """Thread to handle LIDAR data acquisition."""
    global lidar_data
    try:
        print("Starting LIDAR data acquisition...")
        for scan in lidar.iter_scans():
            lidar_data = [
                (item[1], item[2] / 1000.0)  # Convert distance to meters
                for item in scan
                if len(item) >= 3 and (0 <= item[1] <= 20 or 340 <= item[1] <= 360) and 0.5 <= item[2] / 1000.0 <= 3.0
            ]   # item[1] angle / item[2] distance
            print("Filtered LIDAR Data (first 5 points):", lidar_data[:5])
            time.sleep(0.1)
    except Exception as e:
        print("Error in LIDAR thread:", e)
    finally:
        lidar.stop()
        lidar.disconnect()


def rotate_lidar_points(angle, points):
    theta = np.radians(angle)  # Rotation angle
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    rotated_points = []
    for angle, distance in points:
        # Transform LIDAR data in Cartezian coord
        x = distance * np.cos(np.radians(angle))
        y = distance * np.sin(np.radians(angle))
        z = 0
        rotated = np.dot(rotation_matrix, np.array([x, y, z]))
        rotated_points.append(rotated[:2])
    return rotated_points

# Transform
def transform_lidar_to_image(lidar_points, rotation_angle=ROTATION_ANGLE):
    image_points = []
    # Rotate to be aligned with camera orientation
    rotated_points = rotate_lidar_points(rotation_angle, lidar_points)
    for x, y in rotated_points:
        x += LIDAR_TO_CAMERA_X
        y += LIDAR_TO_CAMERA_Y
        z = LIDAR_TO_CAMERA_Z

        if z > 0:
            u = int((FOCAL_LENGTH * x / z) + (IMAGE_WIDTH / 2))
            v = int((FOCAL_LENGTH * y / z) + (IMAGE_HEIGHT / 2))

            # print only valid points that are in image
            u_clamped = max(0, min(u, IMAGE_WIDTH - 1))
            v_clamped = max(0, min(v, IMAGE_HEIGHT - 1))

            if 0 <= u < IMAGE_WIDTH and 0 <= v < IMAGE_HEIGHT:
                image_points.append((u_clamped, v_clamped))
    return image_points


def webcam_thread_with_detection():
    """Thread to handle webcam display with LIDAR overlay and YOLOv8 detection."""
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        global lidar_data
        overlay_points = transform_lidar_to_image(lidar_data)

        # Collect x-coordinates for drawing the rectangle
        x_coords = [point[0] for point in overlay_points]

        if x_coords:
            # Determine the left and right edges of the rectangle
            x_min, x_max = min(x_coords), max(x_coords)

            # Draw the full-height rectangle
            cv2.rectangle(frame, (x_min, 0), (x_max, IMAGE_HEIGHT - 1), (0, 255, 0), 2)  # Green rectangle

        # Overlay LIDAR points on the frame
        for point in overlay_points:
            cv2.circle(frame, point, 5, (255, 0, 0), -1)

        # YOLOv8 person detection
        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []

        for detection in detections:
            x_min, y_min, x_max, y_max, confidence, class_id = detection
            if int(class_id) == 0:  # Check if class_id corresponds to 'person'
                cv2.rectangle(
                    frame,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    (0, 255, 0), 2
                )
                cv2.putText(
                    frame, f"Person: {confidence:.2f}",
                    (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                # Save the frame when a person is detected
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"detected_person_{timestamp}.jpg", frame)
                print(f"Person detected and saved as detected_person_{timestamp}.jpg")

        cv2.imshow("Webcam with LIDAR Overlay and YOLOv8 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main function to start LIDAR and webcam threads."""
    lidar = initialize_lidar()

    lidar_thread_instance = threading.Thread(target=lidar_thread, args=(lidar,))
    webcam_thread_instance = threading.Thread(target=webcam_thread_with_detection)

    lidar_thread_instance.start()
    webcam_thread_instance.start()

    lidar_thread_instance.join()
    webcam_thread_instance.join()


if __name__ == "__main__":
    main()
