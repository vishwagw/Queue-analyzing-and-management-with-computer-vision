from ultralytics import YOLO
import cv2

class QueueDetection:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.queue_threshold = 3  # Minimum people for queue
        
    def detect_people(self, image):
        results = self.model(image)
        people = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == 0:  # person class
                    people.append(box.xyxy[0].cpu().numpy())
        
        return people
    
    def is_queue_present(self, people, queue_zone):
        people_in_queue = self.count_people_in_zone(people, queue_zone)
        return len(people_in_queue) >= self.queue_threshold