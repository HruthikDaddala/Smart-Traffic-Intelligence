import cv2
import torch
from ultralytics import YOLO
import numpy as np

class TrafficAI:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        # Vehicle classes in COCO: 2 (car), 3 (motorcycle/bike), 5 (bus), 7 (truck)
        # We also want to detect ambulance. YOLOv8 doesn't have a specific 'ambulance' class in default COCO.
        # We can detect 'truck' or 'car' and use color or shape or just a mock detection for this specific scenario.
        # However, for a complete system, we could use a custom model.
        # For this requirement, I'll use class 0 (person) as a placeholder for ambulance if needed, or stick to COCO.
        # Let's assume we detect ambulance by detecting car/truck and checking for red/blue flashing lights (mocked here).
        self.vehicle_classes = [2, 3, 5, 7] 

    def get_lanes(self, frame_width, frame_height):
        # Divide frame into 3 vertical lanes (ROIs)
        lane1 = [0, 0, frame_width // 3, frame_height]
        lane2 = [frame_width // 3, 0, 2 * (frame_width // 3), frame_height]
        lane3 = [2 * (frame_width // 3), 0, frame_width, frame_height]
        return [lane1, lane2, lane3]

    def process_frame(self, frame):
        results = self.model(frame)[0]
        detections = []
        lane_counts = [0, 0, 0]
        emergency_detected = False

        frame_height, frame_width = frame.shape[:2]
        lanes = self.get_lanes(frame_width, frame_height)

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if int(class_id) in self.vehicle_classes:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Determine lane
                assigned_lane = -1
                if center_x < lanes[0][2]:
                    assigned_lane = 0
                elif center_x < lanes[1][2]:
                    assigned_lane = 1
                else:
                    assigned_lane = 2
                
                if assigned_lane != -1:
                    lane_counts[assigned_lane] += 1

                # Mock Ambulance Detection
                # In real life, we'd use a custom YOLO model train on ambulances
                # OR use sound or color detection.
                # Here, we'll label a 'bus' or 'truck' as emergency if it's in a certain ROI or has custom properties.
                # For demonstration, let's say class 5 (bus) is treated as an emergency if it's very large (mocking importance).
                is_emergency = False
                if class_id == 5 and score > 0.8: # Example condition
                    emergency_detected = True
                    is_emergency = True

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class": self.model.names[int(class_id)],
                    "confidence": score,
                    "lane": assigned_lane + 1,
                    "is_emergency": is_emergency
                })

        return {
            "detections": detections,
            "lane_counts": lane_counts,
            "total_count": sum(lane_counts),
            "emergency_detected": emergency_detected
        }

    def draw_detections(self, frame, results):
        for det in results["detections"]:
            x1, y1, x2, y2 = det["box"]
            color = (0, 0, 255) if det["is_emergency"] else (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw Lane Dividers
        h, w = frame.shape[:2]
        cv2.line(frame, (w//3, 0), (w//3, h), (255, 255, 0), 2)
        cv2.line(frame, (2*w//3, 0), (2*w//3, h), (255, 255, 0), 2)
        
        return frame
