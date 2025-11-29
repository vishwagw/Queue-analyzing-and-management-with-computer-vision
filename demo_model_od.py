import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import argparse
import os

class RestaurantQueueDetector:
    def __init__(self):
        # Load pre-trained YOLO model for person detection
        self.model = YOLO('yolov8n.pt')
        self.queue_threshold = 3
        self.queue_history = deque(maxlen=30)
        
        # Queue zone will be set based on video dimensions
        self.queue_zone = None
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'queue_frames': 0,
            'max_queue_length': 0,
            'total_people_detected': 0
        }
        
        # Colors
        self.colors = {
            'queue_zone': (0, 255, 255),
            'person': (0, 255, 0),
            'queue_text': (0, 0, 255),
            'info_text': (255, 255, 255),
            'stats_text': (255, 255, 0)
        }
    
    def set_queue_zone(self, frame_width, frame_height):
        """Set queue zone based on video dimensions"""
        self.queue_zone = {
            'x1': int(frame_width * 0.2),
            'y1': int(frame_height * 0.3),
            'x2': int(frame_width * 0.8),
            'y2': int(frame_height * 0.7)
        }
    
    def is_point_in_queue_zone(self, x, y):
        """Check if a point is within the queue zone"""
        if self.queue_zone is None:
            return False
        return (self.queue_zone['x1'] <= x <= self.queue_zone['x2'] and 
                self.queue_zone['y1'] <= y <= self.queue_zone['y2'])
    
    def calculate_wait_time(self, queue_length):
        """Estimate wait time based on queue length"""
        return queue_length * 2
    
    def detect_queue(self, frame):
        """Main function to detect queue in the frame"""
        self.stats['total_frames'] += 1
        
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        people_in_queue = 0
        person_positions = []
        total_people = 0
        
        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == 0 and box.conf > 0.5:
                        total_people += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Calculate center point
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Check if person is in queue zone
                        if self.is_point_in_queue_zone(center_x, center_y):
                            people_in_queue += 1
                            person_positions.append((center_x, center_y))
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                        self.colors['person'], 2)
                            
                            # Draw center point
                            cv2.circle(frame, (center_x, center_y), 5, 
                                      self.colors['person'], -1)
        
        # Update stats
        self.stats['total_people_detected'] += total_people
        if people_in_queue > self.stats['max_queue_length']:
            self.stats['max_queue_length'] = people_in_queue
        
        # Determine queue status
        is_queue = people_in_queue >= self.queue_threshold
        wait_time = self.calculate_wait_time(people_in_queue)
        
        if is_queue:
            self.stats['queue_frames'] += 1
        
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
            'person_positions': person_positions,
            'total_people': total_people
        }
    
    def draw_visualizations(self, frame, queue_data, frame_count, fps):
        """Draw visual elements on the frame"""
        if self.queue_zone:
            # Draw queue zone
            cv2.rectangle(frame, 
                         (self.queue_zone['x1'], self.queue_zone['y1']),
                         (self.queue_zone['x2'], self.queue_zone['y2']),
                         self.colors['queue_zone'], 2)
            
            # Draw queue zone label
            cv2.putText(frame, "QUEUE ZONE", 
                       (self.queue_zone['x1'], self.queue_zone['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       self.colors['queue_zone'], 2)
        
        # Main status information
        status_color = (0, 0, 255) if queue_data['is_queue'] else (0, 255, 0)
        status_text = f"QUEUE DETECTED: {'YES' if queue_data['is_queue'] else 'NO'}"
        
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Queue details
        details_y = 60
        cv2.putText(frame, f"People in queue: {queue_data['queue_length']}", 
                   (10, details_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   self.colors['info_text'], 2)
        
        cv2.putText(frame, f"Estimated wait: {queue_data['wait_time']} min", 
                   (10, details_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   self.colors['info_text'], 2)
        
        cv2.putText(frame, f"Total people in frame: {queue_data['total_people']}", 
                   (10, details_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   self.colors['info_text'], 2)
        
        # Video info
        info_y = details_y + 85
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   self.colors['stats_text'], 1)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   self.colors['stats_text'], 1)
        
        # Queue trend
        trend = self.analyze_queue_trend()
        trend_color = (0, 0, 255) if trend == "Growing" else (0, 255, 0) if trend == "Shrinking" else (255, 255, 0)
        cv2.putText(frame, f"Trend: {trend}", 
                   (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   trend_color, 2)
        
        return frame
    
    def draw_statistics(self, frame):
        """Draw overall statistics on the frame"""
        stats_y = 180
        
        # Calculate percentages
        queue_percentage = (self.stats['queue_frames'] / self.stats['total_frames'] * 100) if self.stats['total_frames'] > 0 else 0
        
        stats_texts = [
            f"STATISTICS:",
            f"Queue detected in: {queue_percentage:.1f}% of frames",
            f"Max queue length: {self.stats['max_queue_length']} people",
            f"Total frames processed: {self.stats['total_frames']}",
            f"Avg people per frame: {self.stats['total_people_detected']/self.stats['total_frames']:.1f}" if self.stats['total_frames'] > 0 else "Avg people per frame: 0"
        ]
        
        for i, text in enumerate(stats_texts):
            color = self.colors['stats_text'] if i > 0 else (255, 255, 255)
            font_scale = 0.5 if i > 0 else 0.6
            thickness = 1 if i > 0 else 2
            cv2.putText(frame, text, (10, stats_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        return frame
    
    def analyze_queue_trend(self):
        """Analyze queue trend over time"""
        if len(self.queue_history) < 5:
            return "Analyzing..."
        
        recent_queues = [entry['queue_length'] for entry in list(self.queue_history)[-10:]]
        
        if np.std(recent_queues) < 0.5:  # Very little variation
            return "Stable"
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_queues))
        slope = np.polyfit(x, recent_queues, 1)[0]
        
        if slope > 0.1:
            return "Growing"
        elif slope < -0.1:
            return "Shrinking"
        else:
            return "Stable"
    
    def get_final_statistics(self):
        """Get final statistics after processing"""
        queue_percentage = (self.stats['queue_frames'] / self.stats['total_frames'] * 100) if self.stats['total_frames'] > 0 else 0
        
        return {
            'total_frames': self.stats['total_frames'],
            'queue_frames': self.stats['queue_frames'],
            'queue_percentage': queue_percentage,
            'max_queue_length': self.stats['max_queue_length'],
            'total_people_detected': self.stats['total_people_detected']
        }

def process_video(input_path, output_path=None):
    """Process a video file for queue detection"""
    
    if not os.path.exists(input_path):
        print(f"Error: Input video file '{input_path}' not found!")
        return
    
    # Initialize detector
    detector = RestaurantQueueDetector()
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set queue zone based on video dimensions
    detector.set_queue_zone(frame_width, frame_height)
    
    # Setup video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    else:
        out = None
    
    print(f"Processing video: {input_path}")
    print(f"Video info: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} frames")
    print("Press 'q' to quit, 'p' to pause, 'r' to reset queue zone")
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect queue in current frame
            start_time = time.time()
            queue_data = detector.detect_queue(frame)
            processing_time = time.time() - start_time
            current_fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Draw visualizations
            frame = detector.draw_visualizations(frame, queue_data, frame_count, current_fps)
            frame = detector.draw_statistics(frame)
            
            # Progress bar
            progress = frame_count / total_frames
            bar_width = 300
            bar_height = 20
            cv2.rectangle(frame, (frame_width//2 - bar_width//2, frame_height - 40), 
                         (frame_width//2 + bar_width//2, frame_height - 20), (50, 50, 50), -1)
            cv2.rectangle(frame, (frame_width//2 - bar_width//2, frame_height - 40), 
                         (frame_width//2 - bar_width//2 + int(bar_width * progress), frame_height - 20), 
                         (0, 255, 0), -1)
            cv2.putText(frame, f"{progress*100:.1f}%", 
                       (frame_width//2 - 20, frame_height - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame to output if specified
            if out:
                out.write(frame)
        
        # Display frame
        cv2.imshow('Restaurant Queue Detection - Video Analysis', frame)
        
        # Handle key presses
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):
            # Reset queue zone
            detector.set_queue_zone(frame_width, frame_height)
            print("Queue zone reset")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    final_stats = detector.get_final_statistics()
    print("\n=== PROCESSING COMPLETE ===")
    print(f"Total frames processed: {final_stats['total_frames']}")
    print(f"Queue detected in: {final_stats['queue_percentage']:.1f}% of frames")
    print(f"Maximum queue length: {final_stats['max_queue_length']} people")
    print(f"Total people detected: {final_stats['total_people_detected']}")
    
    if output_path:
        print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Restaurant Queue Detection on Recorded Videos')
    parser.add_argument('--input', '-i', required=True, help='Input video file path')
    parser.add_argument('--output', '-o', help='Output video file path (optional)')
    
    args = parser.parse_args()
    
    process_video(args.input, args.output)

if __name__ == "__main__":
    # If no command line arguments, use default video or prompt
    try:
        main()
    except:
        # Demo mode with default video path
        print("Running in demo mode...")
        video_path = input("Enter path to your video file: ").strip().strip('"')
        
        if not video_path:
            print("No video path provided. Please provide a video file.")
            exit(1)
            
        output_path = input("Enter output path (optional, press Enter to skip): ").strip().strip('"')
        
        if not output_path:
            output_path = None
            
        process_video(video_path, output_path)