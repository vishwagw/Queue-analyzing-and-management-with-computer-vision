import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

class RestaurantQueueDetector:
    def __init__(self):
        # Load pre-trained YOLO model for person detection
        self.model = YOLO('yolov8n.pt')  # Using nano version for speed
        self.queue_threshold = 3  # Minimum people to consider as queue
        self.queue_history = deque(maxlen=10)  # Store recent queue states
        
        # Define queue area (you can adjust these coordinates)
        self.queue_zone = {
            'x1': 100, 'y1': 200,
            'x2': 600, 'y2': 400
        }
        
        # Colors for visualization
        self.colors = {
            'queue_zone': (0, 255, 255),  # Yellow
            'person': (0, 255, 0),        # Green
            'queue_text': (0, 0, 255),    # Red
            'info_text': (255, 255, 255)  # White
        }
    
    def is_point_in_queue_zone(self, x, y):
        """Check if a point is within the queue zone"""
        return (self.queue_zone['x1'] <= x <= self.queue_zone['x2'] and 
                self.queue_zone['y1'] <= y <= self.queue_zone['y2'])
    
    def calculate_wait_time(self, queue_length):
        """Estimate wait time based on queue length"""
        # Simple estimation: 2 minutes per person in queue
        return queue_length * 2
    
    def detect_queue(self, frame):
        """Main function to detect queue in the frame"""
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        people_in_queue = 0
        person_positions = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Class 0 is 'person' in YOLO
                    if int(box.cls) == 0 and box.conf > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Calculate center point of bounding box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Check if person is in queue zone
                        if self.is_point_in_queue_zone(center_x, center_y):
                            people_in_queue += 1
                            person_positions.append((center_x, center_y))
                            
                            # Draw bounding box for people in queue
                            cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                        self.colors['person'], 2)
        
        # Determine queue status
        is_queue = people_in_queue >= self.queue_threshold
        wait_time = self.calculate_wait_time(people_in_queue)
        
        # Update queue history
        self.queue_history.append({
            'timestamp': time.time(),
            'queue_length': people_in_queue,
            'is_queue': is_queue
        })
        
        return {
            'is_queue': is_queue,
            'queue_length': people_in_queue,
            'wait_time': wait_time,
            'person_positions': person_positions
        }
    
    def draw_visualizations(self, frame, queue_data):
        """Draw visual elements on the frame"""
        # Draw queue zone
        cv2.rectangle(frame, 
                     (self.queue_zone['x1'], self.queue_zone['y1']),
                     (self.queue_zone['x2'], self.queue_zone['y2']),
                     self.colors['queue_zone'], 2)
        
        # Draw queue status text
        status_text = f"Queue: {'YES' if queue_data['is_queue'] else 'NO'}"
        people_text = f"People in queue: {queue_data['queue_length']}"
        wait_text = f"Est. wait: {queue_data['wait_time']} min"
        
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 
                   self.colors['queue_text'], 2)
        cv2.putText(frame, people_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   self.colors['info_text'], 2)
        cv2.putText(frame, wait_text, (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   self.colors['info_text'], 2)
        
        # Draw person count in queue zone
        cv2.putText(frame, f"Queue Zone", 
                   (self.queue_zone['x1'], self.queue_zone['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   self.colors['queue_zone'], 2)
        
        return frame
    
    def analyze_queue_trend(self):
        """Analyze queue trend over time"""
        if len(self.queue_history) < 2:
            return "Insufficient data"
        
        recent_queues = [entry['queue_length'] for entry in list(self.queue_history)[-5:]]
        if len(recent_queues) < 2:
            return "Stable"
        
        # Simple trend analysis
        if recent_queues[-1] > recent_queues[0]:
            return "Growing"
        elif recent_queues[-1] < recent_queues[0]:
            return "Shrinking"
        else:
            return "Stable"

def main():
    # Initialize detector
    detector = RestaurantQueueDetector()
    
    # Initialize video capture (0 for webcam, or file path for video)
    cap = cv2.VideoCapture(0)  # Change to video file path if needed
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting queue detection...")
    print("Press 'q' to quit")
    print("Press 'r' to reset queue zone")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip frame horizontally if using webcam (optional)
        frame = cv2.flip(frame, 1)
        
        # Detect queue
        queue_data = detector.detect_queue(frame)
        
        # Draw visualizations
        frame = detector.draw_visualizations(frame, queue_data)
        
        # Display additional info
        trend = detector.analyze_queue_trend()
        cv2.putText(frame, f"Trend: {trend}", (10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Restaurant Queue Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset queue zone to entire frame temporarily
            h, w = frame.shape[:2]
            detector.queue_zone = {
                'x1': w//4, 'y1': h//4,
                'x2': 3*w//4, 'y2': 3*h//4
            }
            print("Queue zone reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Queue detection stopped")

if __name__ == "__main__":
    main()