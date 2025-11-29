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
        
        # Professional color scheme
        self.colors = {
            'primary': (74, 144, 226),      # Professional blue
            'success': (76, 175, 80),       # Green
            'warning': (255, 152, 0),       # Orange
            'danger': (244, 67, 54),        # Red
            'info': (33, 150, 243),         # Light blue
            'background': (45, 45, 48),     # Dark gray
            'text_primary': (255, 255, 255), # White
            'text_secondary': (189, 189, 189), # Light gray
            'overlay': (0, 0, 0),           # Black for overlays
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
    
    def draw_rounded_rectangle(self, frame, pt1, pt2, color, thickness=2, radius=10):
        """Draw a rounded rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw main rectangles
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw corners
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    
    def draw_label_with_background(self, frame, text, position, bg_color, text_color, 
                                   font_scale=0.6, thickness=2, padding=8):
        """Draw text with a professional background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = position
        
        # Draw background with slight transparency effect
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + baseline + padding),
                     bg_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, 
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + baseline + padding),
                     bg_color, 2)
        
        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        return text_height + baseline + 2 * padding
    
    def draw_info_panel(self, frame, x, y, width, height, title, items, 
                       bg_color=None, title_color=None):
        """Draw a professional information panel"""
        if bg_color is None:
            bg_color = self.colors['background']
        if title_color is None:
            title_color = self.colors['primary']
        
        # Draw panel background with transparency
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), bg_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw panel border
        cv2.rectangle(frame, (x, y), (x + width, y + height), title_color, 2)
        
        # Draw title bar
        cv2.rectangle(frame, (x, y), (x + width, y + 35), title_color, -1)
        cv2.putText(frame, title, (x + 10, y + 23), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text_primary'], 2, cv2.LINE_AA)
        
        # Draw items
        item_y = y + 55
        for item in items:
            cv2.putText(frame, item, (x + 10, item_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_secondary'], 1, cv2.LINE_AA)
            item_y += 25
    
    def detect_queue(self, frame):
        """Main function to detect queue in the frame"""
        self.stats['total_frames'] += 1
        
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        people_in_queue = 0
        person_positions = []
        total_people = 0
        all_detections = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == 0 and box.conf > 0.5:
                        total_people += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf)
                        
                        # Calculate center point
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Check if person is in queue zone
                        in_queue = self.is_point_in_queue_zone(center_x, center_y)
                        
                        if in_queue:
                            people_in_queue += 1
                            person_positions.append((center_x, center_y))
                        
                        all_detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'confidence': confidence,
                            'in_queue': in_queue
                        })
        
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
            'total_people': total_people,
            'detections': all_detections
        }
    
    def draw_professional_bbox(self, frame, detection, person_id):
        """Draw professional bounding box with enhanced styling"""
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        in_queue = detection['in_queue']
        center_x, center_y = detection['center']
        
        # Choose color based on queue status
        if in_queue:
            primary_color = self.colors['success']
            accent_color = self.colors['info']
        else:
            primary_color = self.colors['text_secondary']
            accent_color = self.colors['text_secondary']
        
        # Draw main bounding box with rounded corners
        self.draw_rounded_rectangle(frame, (x1, y1), (x2, y2), primary_color, 2, 8)
        
        # Draw corner accents (L-shapes in corners)
        corner_length = 15
        corner_thickness = 3
        
        # Top-left corner
        cv2.line(frame, (x1, y1 + corner_length), (x1, y1), accent_color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), accent_color, corner_thickness)
        
        # Top-right corner
        cv2.line(frame, (x2, y1 + corner_length), (x2, y1), accent_color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), accent_color, corner_thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x1, y2 - corner_length), (x1, y2), accent_color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), accent_color, corner_thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x2, y2 - corner_length), (x2, y2), accent_color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), accent_color, corner_thickness)
        
        # Draw label with ID and confidence
        label = f"#{person_id}"
        conf_text = f"{confidence*100:.0f}%"
        
        # Draw label background
        label_height = self.draw_label_with_background(
            frame, label, (x1 + 5, y1 - 5), 
            primary_color, self.colors['text_primary'],
            font_scale=0.5, thickness=2, padding=5
        )
        
        # Draw confidence badge
        if in_queue:
            badge_text = "IN QUEUE"
            badge_color = self.colors['success']
        else:
            badge_text = conf_text
            badge_color = self.colors['text_secondary']
        
        self.draw_label_with_background(
            frame, badge_text, (x1 + 5, y2 + 20),
            badge_color, self.colors['text_primary'],
            font_scale=0.4, thickness=1, padding=4
        )
        
        # Draw center point with glow effect
        cv2.circle(frame, (center_x, center_y), 8, (0, 0, 0), -1)
        cv2.circle(frame, (center_x, center_y), 6, primary_color, -1)
        cv2.circle(frame, (center_x, center_y), 3, self.colors['text_primary'], -1)
    
    def draw_visualizations(self, frame, queue_data, frame_count, fps):
        """Draw visual elements on the frame"""
        height, width = frame.shape[:2]
        
        # Draw queue zone with enhanced styling
        if self.queue_zone:
            # Draw semi-transparent overlay for queue zone
            overlay = frame.copy()
            cv2.rectangle(overlay,
                         (self.queue_zone['x1'], self.queue_zone['y1']),
                         (self.queue_zone['x2'], self.queue_zone['y2']),
                         self.colors['info'], -1)
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
            
            # Draw dashed border for queue zone
            dash_length = 20
            gap_length = 10
            
            # Top border
            for x in range(self.queue_zone['x1'], self.queue_zone['x2'], dash_length + gap_length):
                cv2.line(frame, (x, self.queue_zone['y1']), 
                        (min(x + dash_length, self.queue_zone['x2']), self.queue_zone['y1']),
                        self.colors['info'], 3)
            
            # Bottom border
            for x in range(self.queue_zone['x1'], self.queue_zone['x2'], dash_length + gap_length):
                cv2.line(frame, (x, self.queue_zone['y2']), 
                        (min(x + dash_length, self.queue_zone['x2']), self.queue_zone['y2']),
                        self.colors['info'], 3)
            
            # Left border
            for y in range(self.queue_zone['y1'], self.queue_zone['y2'], dash_length + gap_length):
                cv2.line(frame, (self.queue_zone['x1'], y), 
                        (self.queue_zone['x1'], min(y + dash_length, self.queue_zone['y2'])),
                        self.colors['info'], 3)
            
            # Right border
            for y in range(self.queue_zone['y1'], self.queue_zone['y2'], dash_length + gap_length):
                cv2.line(frame, (self.queue_zone['x2'], y), 
                        (self.queue_zone['x2'], min(y + dash_length, self.queue_zone['y2'])),
                        self.colors['info'], 3)
            
            # Draw zone label
            self.draw_label_with_background(
                frame, "QUEUE MONITORING ZONE",
                (self.queue_zone['x1'] + 10, self.queue_zone['y1'] + 25),
                self.colors['info'], self.colors['text_primary'],
                font_scale=0.7, thickness=2, padding=8
            )
        
        # Draw professional bounding boxes
        for idx, detection in enumerate(queue_data['detections'], 1):
            self.draw_professional_bbox(frame, detection, idx)
        
        # Draw main status panel
        status_color = self.colors['danger'] if queue_data['is_queue'] else self.colors['success']
        status_text = "QUEUE DETECTED" if queue_data['is_queue'] else "NO QUEUE"
        
        panel_items = [
            f"Status: {status_text}",
            f"People in Queue: {queue_data['queue_length']}",
            f"Est. Wait Time: {queue_data['wait_time']} min",
            f"Total in Frame: {queue_data['total_people']}"
        ]
        
        self.draw_info_panel(frame, 20, 20, 300, 150, "QUEUE STATUS", 
                            panel_items, title_color=status_color)
        
        # Draw system info panel
        trend = self.analyze_queue_trend()
        trend_color = (self.colors['danger'] if trend == "Growing" 
                      else self.colors['success'] if trend == "Shrinking" 
                      else self.colors['warning'])
        
        system_items = [
            f"Frame: {frame_count}",
            f"FPS: {fps:.1f}",
            f"Trend: {trend}"
        ]
        
        self.draw_info_panel(frame, width - 270, 20, 250, 120, "SYSTEM INFO", 
                            system_items, title_color=self.colors['primary'])
        
        return frame
    
    def draw_statistics(self, frame):
        """Draw overall statistics on the frame"""
        height, width = frame.shape[:2]
        
        # Calculate percentages
        queue_percentage = (self.stats['queue_frames'] / self.stats['total_frames'] * 100) if self.stats['total_frames'] > 0 else 0
        avg_people = self.stats['total_people_detected']/self.stats['total_frames'] if self.stats['total_frames'] > 0 else 0
        
        stats_items = [
            f"Queue Detected: {queue_percentage:.1f}%",
            f"Max Queue: {self.stats['max_queue_length']} people",
            f"Avg People: {avg_people:.1f}",
            f"Total Frames: {self.stats['total_frames']}"
        ]
        
        self.draw_info_panel(frame, 20, height - 150, 280, 130, "ANALYTICS", 
                            stats_items, title_color=self.colors['warning'])
        
        return frame
    
    def draw_progress_bar(self, frame, progress, frame_count, total_frames):
        """Draw professional progress bar"""
        height, width = frame.shape[:2]
        
        bar_width = 400
        bar_height = 8
        bar_x = width // 2 - bar_width // 2
        bar_y = height - 50
        
        # Draw background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bar_x - 10, bar_y - 25), 
                     (bar_x + bar_width + 10, bar_y + bar_height + 15),
                     self.colors['background'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (bar_x - 10, bar_y - 25), 
                     (bar_x + bar_width + 10, bar_y + bar_height + 15),
                     self.colors['primary'], 2)
        
        # Draw progress bar background
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (70, 70, 70), -1)
        
        # Draw progress
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + progress_width, bar_y + bar_height),
                     self.colors['success'], -1)
        
        # Draw percentage text
        progress_text = f"{progress*100:.1f}% ({frame_count}/{total_frames})"
        cv2.putText(frame, progress_text,
                   (bar_x + bar_width // 2 - 80, bar_y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_primary'], 2, cv2.LINE_AA)
    
    def analyze_queue_trend(self):
        """Analyze queue trend over time"""
        if len(self.queue_history) < 5:
            return "Analyzing..."
        
        recent_queues = [entry['queue_length'] for entry in list(self.queue_history)[-10:]]
        
        if np.std(recent_queues) < 0.5:
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
            detector.draw_progress_bar(frame, progress, frame_count, total_frames)
            
            # Write frame to output if specified
            if out:
                out.write(frame)
        
        # Display frame
        cv2.imshow('Restaurant Queue Detection - Professional Analysis', frame)
        
        # Handle key presses
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):
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
    parser = argparse.ArgumentParser(description='Professional Restaurant Queue Detection System')
    parser.add_argument('--input', '-i', required=True, help='Input video file path')
    parser.add_argument('--output', '-o', help='Output video file path (optional)')
    
    args = parser.parse_args()
    
    process_video(args.input, args.output)

if __name__ == "__main__":
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