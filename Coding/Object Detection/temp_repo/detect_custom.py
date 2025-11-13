import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import torch
import os

class RealTimeDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load the trained model
        self.model = YOLO(model_path)
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        
        # Get class names from the model
        self.classes = self.model.names
        print(f"Loaded model with classes: {self.classes}")
        
    def detect_objects(self):
        try:
            while True:
                # Wait for a coherent pair of frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())
                
                # Run YOLO detection
                results = self.model(frame, verbose=False)
                
                # Process results
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                    conf = result.boxes.conf.cpu().numpy()    # Get confidence scores
                    cls = result.boxes.cls.cpu().numpy()      # Get class IDs
                    
                    # Draw bounding boxes and labels
                    for i, (box, c, conf) in enumerate(zip(boxes, cls, conf)):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = self.classes[int(c)]
                        label = f"{class_name} {conf:.2f}"
                        
                        # Draw rectangle and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display the resulting frame
                cv2.imshow('Custom YOLO Detection', frame)
                
                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # Clean up
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Initialize and start detection
    detector = RealTimeDetector()
    print("Starting detection. Press 'q' to quit.")
    detector.detect_objects()
