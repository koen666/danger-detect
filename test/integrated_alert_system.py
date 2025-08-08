import os
import cv2
import time
import argparse
import threading
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from models.video.video_capture import VideoCaptureManager
from models.video.frame_processor import FrameProcessor
from models.detection.object_tracker import ObjectDetectionAndTracking
from models.behavior.behavior_recognition import BehaviorRecognitionSystem
from models.graph.rga import RelationGraphBuilder, SceneGraphBuilder
from models.multimodal.qwen_vl import QwenVLFeatureExtractor
from models.motion.motion_features import MotionFeatureManager
from models.alert.alert_system import AlertSystem


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Integrated Alert System Application")
    
    # Input sources
    parser.add_argument("--source", type=str, default="0", help="Video source (camera index, file path, or URL)")
    
    # Video parameters
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    
    # Module options
    parser.add_argument("--enable_all", action="store_true", help="Enable all modules")
    parser.add_argument("--enable_detection", action="store_true", help="Enable object detection and tracking")
    parser.add_argument("--enable_behavior", action="store_true", help="Enable behavior recognition")
    parser.add_argument("--enable_vl", action="store_true", help="Enable Qwen-VL multimodal features")
    parser.add_argument("--enable_rga", action="store_true", help="Enable RGA relationship modeling")
    parser.add_argument("--enable_motion", action="store_true", help="Enable motion feature extraction")
    
    # Configuration
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="alert_output", help="Output directory for alerts")
    parser.add_argument("--vl_model", type=str, default="Qwen/Qwen-VL-Chat", help="Qwen-VL model version")
    parser.add_argument("--detector", type=str, default="yolov4", help="Object detector model (yolov4, ssd_mobilenet)")
    
    # Alert system options
    parser.add_argument("--rules_file", type=str, default=None, help="Path to alert rules file")
    parser.add_argument("--alert_interval", type=int, default=30, help="Minimum interval between alerts (seconds)")
    
    # Display options
    parser.add_argument("--display", action="store_true", help="Display video output")
    parser.add_argument("--fullscreen", action="store_true", help="Display in fullscreen mode")
    
    return parser.parse_args()


class IntegratedAlertSystem:
    """
    Integrated alert system application that combines all modules:
    - Video capture
    - Object detection and tracking
    - Behavior recognition
    - RGA relationship modeling
    - Qwen-VL multimodal features
    - Motion feature extraction
    - Alert generation and notification
    """
    
    def __init__(self, args):
        """
        Initialize the integrated alert system.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        
        # Set output directory
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize video capture
        self.video_capture = VideoCaptureManager(
            source=args.source,
            target_width=args.width,
            target_height=args.height,
            target_fps=args.fps
        )
        
        # Initialize frame processor (for basic processing)
        self.frame_processor = FrameProcessor()
        
        # Initialize enabled modules
        self.init_modules()
        
        # Initialize alert system
        alert_config_path = args.config
        if args.rules_file:
            # Create minimal config with custom rules file
            alert_config = {
                "output_dir": self.output_dir,
                "rules_file": args.rules_file,
                "processing": {
                    "process_every_n_frames": 5
                }
            }
            
            # Save temporary config
            tmp_config_path = os.path.join(self.output_dir, "alert_config_tmp.json")
            with open(tmp_config_path, 'w') as f:
                json.dump(alert_config, f, indent=2)
            
            alert_config_path = tmp_config_path
        
        self.alert_system = AlertSystem(alert_config_path)
        
        # Start the alert system
        self.alert_system.start()
        
        # Initialize state
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        self.latest_alerts = []
        
        # Initialize display
        if args.display:
            cv2.namedWindow("Integrated Alert System", 
                            cv2.WND_PROP_FULLSCREEN if args.fullscreen else cv2.WINDOW_NORMAL)
            if args.fullscreen:
                cv2.setWindowProperty("Integrated Alert System", 
                                      cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def init_modules(self):
        """Initialize processing modules based on arguments."""
        # Parse enable flags
        enable_detection = self.args.enable_all or self.args.enable_detection
        enable_behavior = self.args.enable_all or self.args.enable_behavior
        enable_vl = self.args.enable_all or self.args.enable_vl
        enable_rga = self.args.enable_all or self.args.enable_rga
        enable_motion = self.args.enable_all or self.args.enable_motion
        
        # Initialize object detection and tracking
        self.object_tracker = None
        if enable_detection:
            self.object_tracker = ObjectDetectionAndTracking(
                detector_type=self.args.detector,
                confidence_threshold=0.5
            )
        
        # Initialize behavior recognition
        self.behavior_recognition = None
        if enable_behavior and enable_detection:  # Behavior needs detection
            self.behavior_recognition = BehaviorRecognitionSystem(
                frame_width=self.args.width,
                frame_height=self.args.height
            )
        
        # Initialize Qwen-VL
        self.qwen_vl = None
        if enable_vl:
            self.qwen_vl = QwenVLFeatureExtractor(
                model_version=self.args.vl_model
            )
        
        # Initialize RGA
        self.relation_graph_builder = None
        self.scene_graph_builder = None
        if enable_rga and enable_vl:  # RGA needs Qwen-VL
            self.relation_graph_builder = RelationGraphBuilder()
            self.scene_graph_builder = SceneGraphBuilder()
        
        # Initialize motion features
        self.motion_manager = None
        if enable_motion:
            self.motion_manager = MotionFeatureManager(
                use_optical_flow=True,
                use_motion_history=True,
                use_gpu=False  # Set to True for GPU acceleration
            )
    
    def run(self):
        """Run the integrated alert system application."""
        if not self.video_capture.is_opened():
            print("Failed to open video source")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        print(f"Starting integrated alert system with source: {self.args.source}")
        
        prev_frame = None
        
        try:
            while self.is_running and self.video_capture.is_opened():
                # Read frame
                ret, frame = self.video_capture.read()
                if not ret or frame is None:
                    print("End of video or error reading frame")
                    break
                
                # Skip frames to meet target FPS if needed
                elapsed = time.time() - self.start_time
                expected_frames = int(elapsed * self.args.fps)
                if self.frame_count < expected_frames:
                    self.frame_count += 1
                    continue
                
                # Process frame
                processed_data = self.process_frame(frame, prev_frame)
                
                # Generate alerts
                if processed_data:
                    alerts = self.alert_system.process_frame(
                        frame_idx=self.frame_count,
                        frame=frame,
                        behavior_results=processed_data.get("behavior_results"),
                        tracks=processed_data.get("tracks"),
                        motion_features=processed_data.get("motion_features"),
                        scene_data=processed_data.get("scene_data"),
                        process_now=True
                    )
                    
                    if alerts:
                        self.latest_alerts = alerts
                
                # Display frame if requested
                if self.args.display:
                    self.display_frame(frame, processed_data)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):  # ESC or q to quit
                        break
                
                # Update frame count and previous frame
                self.frame_count += 1
                prev_frame = frame.copy()
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            self.cleanup()
    
    def process_frame(self, frame, prev_frame):
        """
        Process a frame with all enabled modules.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame
            
        Returns:
            Dictionary with processing results
        """
        processed_data = {}
        
        # Track objects
        tracks = None
        detections = None
        if self.object_tracker:
            detections, tracks, _, _ = self.object_tracker.process_frame(frame)
            processed_data["detections"] = detections
            processed_data["tracks"] = tracks
        
        # Extract motion features
        motion_features = None
        if self.motion_manager:
            motion_features = self.motion_manager.extract_features(frame, tracks=tracks)
            processed_data["motion_features"] = motion_features
        
        # Analyze behavior
        behavior_results = None
        if self.behavior_recognition and tracks:
            behavior_results = self.behavior_recognition.update(tracks, motion_features or [])
            processed_data["behavior_results"] = behavior_results
        
        # Generate captions and extract VL features
        scene_data = None
        if self.qwen_vl and self.frame_count % 30 == 0:  # Process every 30 frames to reduce load
            caption = self.qwen_vl.generate_caption(frame)
            vl_features = self.qwen_vl.extract_features(frame)
            
            scene_data = {
                "caption": caption,
                "features": vl_features
            }
            
            # Check for anomalies if enabled
            if self.frame_count % 90 == 0:  # Check less frequently
                normal_context = "people walking normally in a hallway"
                anomaly_score = self.qwen_vl.detect_anomalies(frame, normal_context)
                if anomaly_score is not None:
                    scene_data["anomaly_score"] = anomaly_score
            
            processed_data["scene_data"] = scene_data
            
            # Generate relationship graph
            if self.relation_graph_builder and vl_features is not None:
                # Build relationship graph
                adjacency_matrix, graph_features = self.relation_graph_builder.build_graph(vl_features)
                processed_data["relation_graph"] = (adjacency_matrix, graph_features)
                
                # Build scene graph
                if caption:
                    scene_graph = self.scene_graph_builder.build_from_caption(caption, vl_features)
                    processed_data["scene_graph"] = scene_graph
        
        return processed_data
    
    def display_frame(self, frame, processed_data):
        """
        Display frame with overlays.
        
        Args:
            frame: Current frame
            processed_data: Processing results
        """
        display_frame = frame.copy()
        
        # Draw object tracking results
        if self.object_tracker and "tracks" in processed_data:
            display_frame = self.object_tracker.draw_results(
                display_frame, 
                tracks=processed_data["tracks"]
            )
        
        # Draw behavior results
        if self.behavior_recognition and "behavior_results" in processed_data:
            display_frame = self.behavior_recognition.visualize_behaviors(
                display_frame,
                processed_data["behavior_results"]
            )
        
        # Draw motion features
        if self.motion_manager and "motion_features" in processed_data:
            display_frame = self.motion_manager.visualize_features(
                display_frame,
                processed_data["motion_features"]
            )
        
        # Draw alerts
        self.draw_alerts(display_frame)
        
        # Draw caption
        if "scene_data" in processed_data and "caption" in processed_data["scene_data"]:
            caption = processed_data["scene_data"]["caption"]
            cv2.putText(display_frame, caption[:100], (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw anomaly score if available
            if "anomaly_score" in processed_data["scene_data"]:
                anomaly_score = processed_data["scene_data"]["anomaly_score"]
                color = (0, 255, 0) if anomaly_score < 0.7 else (0, 0, 255)
                cv2.putText(display_frame, f"Anomaly: {anomaly_score:.2f}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw FPS
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Integrated Alert System", display_frame)
    
    def draw_alerts(self, frame):
        """Draw recent alerts on the frame."""
        if not self.latest_alerts:
            return
        
        # Draw alert box in top-right corner
        h, w = frame.shape[:2]
        alert_box_width = min(400, w // 2)
        alert_box_height = min(160, h // 3)
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - alert_box_width, 0), (w, alert_box_height), 
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw alerts (most recent at the top)
        cv2.putText(frame, "ALERTS", (w - alert_box_width + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos = 60
        for i, alert in enumerate(self.latest_alerts[:3]):  # Show max 3 alerts
            level_color = {
                "INFO": (255, 255, 0),    # Cyan
                "WARNING": (0, 165, 255),  # Orange
                "ALERT": (0, 0, 255),     # Red
                "CRITICAL": (0, 0, 255)   # Red
            }.get(alert.level.name, (255, 255, 255))
            
            # Draw timestamp and message
            timestamp = datetime.fromtimestamp(alert.timestamp).strftime('%H:%M:%S')
            text = f"{timestamp} - {alert.message[:40]}"
            cv2.putText(frame, text, (w - alert_box_width + 10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, level_color, 1)
            y_pos += 30
    
    def cleanup(self):
        """Clean up resources."""
        print("Shutting down integrated alert system")
        
        # Stop alert system
        self.alert_system.stop()
        
        # Release video capture
        self.video_capture.release()
        
        # Close display windows
        if self.args.display:
            cv2.destroyAllWindows()
        
        print(f"Processed {self.frame_count} frames")
        print(f"Alert statistics: {self.alert_system.get_alert_stats()}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Enable all modules if --enable_all is specified
    if args.enable_all:
        args.enable_detection = True
        args.enable_behavior = True
        args.enable_vl = True
        args.enable_rga = True
        args.enable_motion = True
    
    # Initialize and run the system
    system = IntegratedAlertSystem(args)
    system.run()


if __name__ == "__main__":
    main() 
