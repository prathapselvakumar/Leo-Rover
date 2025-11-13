import cv2
from ultralytics import YOLO
import numpy as np
import argparse

# ----------------------- COLOR DETECTION FUNCTION -----------------------
def get_dominant_color(image):
    """Return dominant color name (basic color classification)."""
    # Convert image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Compute average color
    avg_hue = np.mean(hsv[:, :, 0])
    avg_sat = np.mean(hsv[:, :, 1])
    avg_val = np.mean(hsv[:, :, 2])

    # Simple HSV-based color classification
    if avg_sat < 40 and avg_val > 200:
        return "White"
    elif avg_val < 50:
        return "Black"
    elif avg_sat < 40:
        return "Gray"
    elif avg_hue < 10 or avg_hue > 170:
        return "Red"
    elif 10 <= avg_hue < 25:
        return "Orange"
    elif 25 <= avg_hue < 35:
        return "Yellow"
    elif 35 <= avg_hue < 85:
        return "Green"
    elif 85 <= avg_hue < 125:
        return "Blue"
    elif 125 <= avg_hue < 160:
        return "Purple"
    else:
        return "Unknown"

# ----------------------- DETECTION FUNCTION -----------------------
def detect_objects_with_color(model, frame, confidence=0.5):
    """Detect objects and annotate with class name + color"""
    results = model(frame, conf=confidence)
    annotated = frame.copy()

    for result in results:
        boxes = result.boxes
        names = model.names

        for box in boxes:
            cls = int(box.cls[0])
            label = names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop object region
            obj_crop = frame[y1:y2, x1:x2]
            if obj_crop.size == 0:
                continue

            # Get dominant color
            color_name = get_dominant_color(obj_crop)
            display_text = f"{label} - {color_name}"

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return annotated

# ----------------------- IMAGE PROCESSING -----------------------
def process_image(model, image_path, confidence=0.5):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image at {image_path}")
        return

    annotated = detect_objects_with_color(model, frame, confidence)
    cv2.imshow('YOLOv8 Object + Color Detection (Image)', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----------------------- WEBCAM PROCESSING -----------------------
def process_webcam(model, confidence=0.5, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Starting YOLOv8 Object + Color Detection...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting...")
            break

        annotated = detect_objects_with_color(model, frame, confidence)
        cv2.imshow('YOLOv8 Object + Color Detection (Webcam)', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------- MAIN -----------------------
def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Object + Color Detection')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0-1)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--camera', type=int, default=2, help='Camera index (default: 2 for RealSense RGB)')
    args = parser.parse_args()

    try:
        model = YOLO(args.model)
        print(f"✅ Model loaded: {args.model}")
        print(f"📸 Using camera index: {args.camera}")
        print(f"🎯 Confidence threshold: {args.confidence}")

        if args.image:
            process_image(model, args.image, args.confidence)
        else:
            process_webcam(model, args.confidence, args.camera)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

