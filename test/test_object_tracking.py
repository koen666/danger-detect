import cv2
import numpy as np
import argparse
import time
import os
from datetime import datetime

from models.video_capture import VideoCaptureManager
from models.object_tracker import ObjectDetectionAndTracking


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Object Detection and Tracking Test')
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source (camera index or file path)'
    )
    parser.add_argument(
        '--detector',
        type=str,
        choices=['yolov4', 'ssd_mobilenet', 'faster_rcnn'],
        default='yolov4',
        help='Detector type'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='tracking_results',
        help='Directory to save output results'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=0,
        help='Save frame every N seconds (0 for manual only)'
    )
    return parser.parse_args()


class ObjectTrackingDemo:
    """Demo application for object detection and tracking."""
    
    def __init__(self, args):
        """
        Initialize the demo.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.source = int(args.source) if args.source.isdigit() else args.source
        self.detector_type = args.detector
        self.confidence = args.confidence
        self.output_dir = args.output_dir
        self.save_interval = args.save_interval
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize video capture
        self.capture_manager = VideoCaptureManager()
        
        # Initialize object detection and tracking
        self.tracker = ObjectDetectionAndTracking(
            detector_type=self.detector_type,
            confidence_threshold=self.confidence
        )
        
        # State variables
        self.is_paused = False
        self.show_help = False
        self.show_detections = True
        self.last_saved_time = 0
    
    def run(self):
        """Run the demo."""
        # Connect to video source
        if not self.capture_manager.open_source(self.source):
            print(f"Failed to open source: {self.source}")
            return
        
        print(f"Running object tracking with {self.detector_type} detector")
        print("Controls:")
        print("  q/ESC: Quit")
        print("  p: Pause/Resume")
        print("  h: Toggle help overlay")
        print("  d: Toggle detection display")
        print("  s: Save current frame")
        print("  r: Reset tracker")
        
        # Main loop
        while True:
            # Read frame
            if self.is_paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
                self._handle_keypress(key)
                continue
            
            success, frame, frame_num = next(self.capture_manager.read_frames())
            if not success:
                print("Failed to read frame")
                break
            
            # Process frame
            detections, tracks, det_time, track_time = self.tracker.process_frame(frame)
            
            # Draw results
            if self.show_detections:
                result_frame = self.tracker.draw_results(frame, detections, tracks)
            else:
                result_frame = self.tracker.draw_results(frame, None, tracks)
            
            # Add UI elements
            self._add_ui_elements(result_frame, len(detections), len(tracks))
            
            # Display frame
            cv2.imshow("Object Tracking", result_frame)
            
            # Check for auto-save
            if self.save_interval > 0:
                current_time = time.time()
                if current_time - self.last_saved_time >= self.save_interval:
                    self._save_frame(result_frame)
                    self.last_saved_time = current_time
            
            # Handle keypress
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            self._handle_keypress(key)
        
        # Clean up
        self.capture_manager.close()
        cv2.destroyAllWindows()
    
    def _add_ui_elements(self, frame, num_detections, num_tracks):
        """
        Add UI elements to the frame.
        
        Args:
            frame: Input frame
            num_detections: Number of detections
            num_tracks: Number of tracks
        """
        h, w = frame.shape[:2]
        
        # Add semi-transparent overlay at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add status text
        status = f"Detector: {self.detector_type.upper()} | Detections: {num_detections} | Tracks: {num_tracks}"
        cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add help overlay if enabled
        if self.show_help:
            self._add_help_overlay(frame)
    
    def _add_help_overlay(self, frame):
        """
        Add help overlay to the frame.
        
        Args:
            frame: Input frame
        """
        h, w = frame.shape[:2]
        
        # Add semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add help text
        help_text = [
            "CONTROLS:",
            "q/ESC: Quit",
            "p: Pause/Resume",
            "h: Toggle help overlay",
            "d: Toggle detection display",
            "s: Save current frame",
            "r: Reset tracker"
        ]
        
        for i, text in enumerate(help_text):
            cv2.putText(frame, text, (20, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Add performance metrics
        metrics = self.tracker.get_performance_metrics()
        metrics_text = [
            f"PERFORMANCE:",
            f"Detection time: {metrics['avg_detection_time']:.4f}s",
            f"Tracking time: {metrics['avg_tracking_time']:.4f}s",
            f"Total time: {metrics['avg_total_time']:.4f}s",
            f"Frames: {metrics['frame_count']}"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(frame, text, (w - 300, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    def _handle_keypress(self, key):
        """
        Handle keyboard input.
        
        Args:
            key: Key code
        """
        if key == ord('p'):  # Pause/Resume
            self.is_paused = not self.is_paused
            print(f"{'Paused' if self.is_paused else 'Resumed'}")
        
        elif key == ord('h'):  # Toggle help
            self.show_help = not self.show_help
        
        elif key == ord('d'):  # Toggle detection display
            self.show_detections = not self.show_detections
            print(f"Detection display: {'ON' if self.show_detections else 'OFF'}")
        
        elif key == ord('s'):  # Save frame
            _, frame, _ = next(self.capture_manager.read_frames())
            detections, tracks, _, _ = self.tracker.process_frame(frame)
            
            if self.show_detections:
                result_frame = self.tracker.draw_results(frame, detections, tracks)
            else:
                result_frame = self.tracker.draw_results(frame, None, tracks)
            
            self._add_ui_elements(result_frame, len(detections), len(tracks))
            self._save_frame(result_frame)
        
        elif key == ord('r'):  # Reset tracker
            self.tracker.reset()
            print("Tracker reset")
    
    def _save_frame(self, frame):
        """
        Save the current frame.
        
        Args:
            frame: Frame to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"tracking_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved frame to {filename}")


def main():
    """Main entry point."""
    args = parse_args()
    demo = ObjectTrackingDemo(args)
    demo.run()


if __name__ == "__main__":
    main() 