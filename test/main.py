#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
24-hour Video Analysis Alert System - Main Program
Main application integrating all modules to provide complete video analysis, behavior recognition, and alert functionality.
"""

import os
import sys
import time
import argparse
import threading
import logging
import cv2
import numpy as np
from datetime import datetime

# Video capture and processing
from models.video.video_capture import VideoCaptureManager
from models.video.frame_processor import FrameProcessor

# Object detection and tracking
from models.object_detection.object_tracker import ObjectTracker, Detection

# Motion feature extraction
from models.motion.motion_manager import MotionFeatureManager
from models.motion.optical_flow import OpticalFlowExtractor
from models.motion.motion_history import MotionHistoryExtractor

# Trajectory analysis
from models.trajectory.trajectory_manager import TrajectoryManager
from models.trajectory.interaction_detector import InteractionDetector

# Behavior recognition
from models.behavior.behavior_recognition import BehaviorRecognizer
from models.behavior.behavior_types import BehaviorType, Interaction

# Visual language model
from models.visual_language.qwen_vl import QwenVLFeatureExtractor

# Graph models
from models.visual_language.rga import SceneGraphBuilder, RelationGraphBuilder

# Alert system
from models.alert.rule_analyzer import RuleAnalyzer
from models.alert.alert_processor import AlertProcessor
from models.alert.notification_manager import NotificationManager

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("video_analysis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class VideoAnalysisSystem:
    """Video analysis system integrating all modules"""
    
    def __init__(self, args):
        """Initialize video analysis system"""
        self.args = args
        self.running = False
        self.paused = False
        self.show_help = False
        self.frame_count = 0
        self.last_alert_time = time.time()
        self.result_dir = args.output_dir
        
        # Ensure output directory exists
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Initialize video capture manager
        self.capture_manager = VideoCaptureManager(
            source=args.source,
            width=args.width,
            height=args.height,
            fps=args.fps
        )
        
        # Initialize frame processor
        self.frame_processor = FrameProcessor(
            preprocessing_config={
                'denoise': args.denoise,
                'enhance': args.enhance,
                'resize_dim': (args.width, args.height)
            }
        )
        
        # Initialize object detector and tracker
        self.object_tracker = ObjectTracker(
            confidence_threshold=args.confidence,
            nms_threshold=args.nms_threshold,
            use_gpu=args.use_gpu
        )
        
        # Initialize motion feature manager
        self.motion_manager = MotionFeatureManager(
            use_optical_flow=args.use_optical_flow,
            use_motion_history=args.use_motion_history,
            use_gpu=args.use_gpu
        )
        
        # Initialize trajectory manager
        self.trajectory_manager = TrajectoryManager(
            max_trajectory_length=args.trajectory_length,
            max_disappeared=args.max_disappeared,
            interaction_distance=args.interaction_threshold
        )
        
        # Initialize behavior recognizer
        self.behavior_recognizer = BehaviorRecognizer(
            speed_threshold=args.speed_threshold,
            interaction_threshold=args.interaction_threshold
        )
        
        # Conditionally initialize visual language model
        self.qwen_vl = None
        self.scene_graph_builder = None
        if args.enable_vl:
            try:
                self.qwen_vl = QwenVLFeatureExtractor(
                    model_version=args.model_version,
                    device=args.device
                )
                self.scene_graph_builder = SceneGraphBuilder(
                    num_entities=args.num_entities
                )
                logger.info("Visual language model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize visual language model: {str(e)}")
        
        # Initialize alert system
        self.rule_analyzer = RuleAnalyzer(
            rules_config=args.rules_config
        )
        self.alert_processor = AlertProcessor(
            min_alert_interval=args.alert_interval
        )
        self.notification_manager = NotificationManager(
            notification_config=args.notification_config
        )
        
        # Initialize threads and locks
        self.frame_lock = threading.Lock()
        self.current_frame = None
        self.processed_frame = None
        self.results = {}
        
        # State dictionary
        self.states = {
            'detected_objects': [],
            'trajectories': [],
            'behaviors': [],
            'interactions': [],
            'scene_graph': None,
            'alerts': []
        }
        
        logger.info("Video analysis system initialized")
    
    def connect(self):
        """Connect to video source"""
        return self.capture_manager.connect()
    
    def process_frame(self, frame):
        """Main pipeline for processing a single frame"""
        # Frame processing (denoising, enhancement, etc.)
        processed_frame = self.frame_processor.process_frame(frame)
        
        # Object detection and tracking
        detections = self.object_tracker.detect(processed_frame)
        tracked_objects = self.object_tracker.update(detections)
        
        # Update trajectories
        self.trajectory_manager.update(tracked_objects)
        trajectories = self.trajectory_manager.get_active_trajectories()
        
        # Extract motion features
        motion_features = self.motion_manager.extract_features(processed_frame, self.capture_manager.get_previous_frame())
        
        # Behavior recognition
        behaviors, interactions = self.behavior_recognizer.analyze(
            trajectories, 
            motion_features,
            self.trajectory_manager.get_interaction_detector()
        )
        
        # Conditional visual language processing
        scene_graph = None
        if self.args.enable_vl and self.qwen_vl is not None and self.frame_count % self.args.vl_interval == 0:
            try:
                caption = self.qwen_vl.generate_caption(processed_frame)
                scene_graph = self.scene_graph_builder.build_scene_graph_from_caption(
                    processed_frame, caption, self.qwen_vl
                )
                logger.info(f"Generated caption: {caption}")
            except Exception as e:
                logger.error(f"Error in visual language processing: {str(e)}")
        
        # Rule analysis and alert processing
        alerts = []
        try:
            alerts = self.rule_analyzer.analyze(
                trajectories=trajectories,
                behaviors=behaviors,
                interactions=interactions,
                scene_graph=scene_graph
            )
        except Exception as e:
            logger.error(f"Error in rule analysis: {str(e)}")
        
        # Update states
        self.states.update({
            'detected_objects': tracked_objects,
            'trajectories': trajectories,
            'behaviors': behaviors,
            'interactions': interactions,
            'scene_graph': scene_graph,
            'alerts': alerts
        })
        
        # Process alerts
        current_time = time.time()
        if alerts and (current_time - self.last_alert_time) > self.args.alert_interval:
            for alert in alerts:
                try:
                    self.alert_processor.process(alert)
                    self.notification_manager.send_notification(alert)
                except Exception as e:
                    logger.error(f"Error processing alert: {str(e)}")
            self.last_alert_time = current_time
            
        # Visualize results
        result_frame = self.visualize_results(processed_frame)
        
        # Update counter
        self.frame_count += 1
        return result_frame
    
    def visualize_results(self, frame):
        """Visualize detection results"""
        visualization_frame = frame.copy()
        
        # Visualize detected objects and trajectories
        for obj in self.states['detected_objects']:
            label = f"{obj.class_name} ({obj.id})"
            color = (0, 255, 0)  # Default green
            
            # Check for associated behaviors and adjust color
            for behavior in self.states['behaviors']:
                if behavior.object_id == obj.id:
                    if behavior.behavior_type == BehaviorType.RUNNING:
                        color = (0, 0, 255)  # Red
                    elif behavior.behavior_type == BehaviorType.LOITERING:
                        color = (255, 0, 0)  # Blue
            
            # Draw bounding box
            cv2.rectangle(
                visualization_frame, 
                (int(obj.x1), int(obj.y1)), 
                (int(obj.x2), int(obj.y2)), 
                color, 
                2
            )
            
            # Draw label
            cv2.putText(
                visualization_frame,
                label,
                (int(obj.x1), int(obj.y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # Draw trajectory if available
            for trajectory in self.states['trajectories']:
                if trajectory.object_id == obj.id:
                    points = trajectory.points
                    for i in range(1, len(points)):
                        # Draw trajectory line
                        cv2.line(
                            visualization_frame,
                            (int(points[i-1][0]), int(points[i-1][1])),
                            (int(points[i][0]), int(points[i][1])),
                            color,
                            2
                        )
        
        # Visualize interactions
        for interaction in self.states['interactions']:
            if interaction.object1_id is not None and interaction.object2_id is not None:
                # Find object positions
                obj1_pos = None
                obj2_pos = None
                
                for obj in self.states['detected_objects']:
                    if obj.id == interaction.object1_id:
                        obj1_pos = (int((obj.x1 + obj.x2) / 2), int((obj.y1 + obj.y2) / 2))
                    elif obj.id == interaction.object2_id:
                        obj2_pos = (int((obj.x1 + obj.x2) / 2), int((obj.y1 + obj.y2) / 2))
                
                if obj1_pos and obj2_pos:
                    # Draw interaction line
                    cv2.line(
                        visualization_frame,
                        obj1_pos,
                        obj2_pos,
                        (255, 255, 0),  # Yellow
                        2
                    )
                    
                    # Add interaction label
                    mid_point = (
                        int((obj1_pos[0] + obj2_pos[0]) / 2),
                        int((obj1_pos[1] + obj2_pos[1]) / 2)
                    )
                    
                    cv2.putText(
                        visualization_frame,
                        interaction.behavior_type.name,
                        mid_point,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),  # Yellow
                        2
                    )
        
        # Add frame info
        fps = self.capture_manager.get_fps()
        frame_info = f"Frame: {self.frame_count} | FPS: {fps:.1f}"
        
        cv2.putText(
            visualization_frame,
            frame_info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),  # Yellow
            2
        )
        
        # Add alerts if any
        if self.states['alerts']:
            alert_text = "ALERTS: " + ", ".join([alert['rule_name'] for alert in self.states['alerts']])
            cv2.putText(
                visualization_frame,
                alert_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red
                2
            )
        
        # Add scene caption if available
        if self.states['scene_graph'] and hasattr(self.states['scene_graph'], 'caption'):
            caption = self.states['scene_graph'].caption
            # Truncate caption if too long
            if len(caption) > 70:
                caption = caption[:67] + "..."
                
            cv2.putText(
                visualization_frame,
                caption,
                (10, visualization_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White
                1
            )
            
        return visualization_frame
    
    def run(self):
        """Run the video analysis system"""
        if not self.connect():
            logger.error("Failed to connect to video source")
            return False
        
        self.running = True
        
        # Main processing loop
        try:
            while self.running:
                # Read frame
                ret, frame = self.capture_manager.read()
                
                if not ret:
                    logger.warning("End of video stream reached or frame reading failed")
                    break
                
                # Skip processing if paused
                if self.paused:
                    # Just display the current processed frame
                    if self.processed_frame is not None:
                        cv2.imshow('Video Analysis', self.processed_frame)
                else:
                    # Process frame
                    try:
                        with self.frame_lock:
                            self.current_frame = frame.copy()
                            self.processed_frame = self.process_frame(frame)
                        
                        # Display the processed frame
                        cv2.imshow('Video Analysis', self.processed_frame)
                        
                        # Save frame if enabled
                        if self.args.save_frames and self.frame_count % self.args.save_interval == 0:
                            self.save_frame(self.processed_frame)
                    except Exception as e:
                        logger.error(f"Error processing frame {self.frame_count}: {str(e)}")
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # Quit
                    self.running = False
                elif key == ord('p'):  # Pause/Resume
                    self.paused = not self.paused
                    logger.info(f"{'Paused' if self.paused else 'Resumed'} video processing")
                elif key == ord('s'):  # Save current frame
                    self.save_frame(self.processed_frame)
                    logger.info(f"Saved frame {self.frame_count}")
                elif key == ord('h'):  # Toggle help
                    self.show_help = not self.show_help
                    
                if self.show_help:
                    self.display_help()
        
        except KeyboardInterrupt:
            logger.info("Video analysis terminated by user")
        except Exception as e:
            logger.error(f"Error in video analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            # Clean up
            self.capture_manager.release()
            cv2.destroyAllWindows()
            
        return True
    
    def save_frame(self, frame):
        """Save current frame to disk"""
        if frame is None:
            return
            
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.result_dir}/frame_{self.frame_count}_{timestamp}.jpg"
        
        # Save frame
        cv2.imwrite(filename, frame)
        logger.info(f"Saved frame to {filename}")
    
    def display_help(self):
        """Display help overlay on video"""
        if self.processed_frame is None:
            return
            
        help_frame = self.processed_frame.copy()
        
        # Create semi-transparent overlay
        overlay = help_frame.copy()
        cv2.rectangle(overlay, (50, 50), (help_frame.shape[1]-50, help_frame.shape[0]-50), (0, 0, 0), -1)
        alpha = 0.7
        help_frame = cv2.addWeighted(overlay, alpha, help_frame, 1-alpha, 0)
        
        # Add help text
        help_text = [
            "Video Analysis System - Help",
            "",
            "Key Controls:",
            "q - Quit the application",
            "p - Pause/Resume video processing",
            "s - Save current frame",
            "h - Toggle this help overlay",
            "",
            "Status:",
            f"Running: {self.running}",
            f"Paused: {self.paused}",
            f"Frame Count: {self.frame_count}",
            f"Optical Flow: {'Enabled' if self.args.use_optical_flow else 'Disabled'}",
            f"Motion History: {'Enabled' if self.args.use_motion_history else 'Disabled'}",
            f"GPU Acceleration: {'Enabled' if self.args.use_gpu else 'Disabled'}"
        ]
        
        y = 100
        for line in help_text:
            cv2.putText(
                help_frame,
                line,
                (100, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            y += 30
        
        cv2.imshow('Video Analysis', help_frame)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="24-hour Video Analysis Alert System")
    
    # Video source options
    parser.add_argument("--source", type=str, default="0",
                       help="Video source (0 for webcam, path to video file, or RTSP URL)")
    parser.add_argument("--width", type=int, default=640,
                       help="Desired width of the video stream")
    parser.add_argument("--height", type=int, default=480,
                       help="Desired height of the video stream")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Desired FPS of the video stream")
    
    # Processing options
    parser.add_argument("--denoise", action="store_true",
                       help="Apply denoising to frames")
    parser.add_argument("--enhance", action="store_true",
                       help="Apply contrast enhancement to frames")
    parser.add_argument("--use_gpu", action="store_true",
                       help="Use GPU acceleration if available")
    
    # Detection options
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold for object detection")
    parser.add_argument("--nms_threshold", type=float, default=0.4,
                       help="Non-maximum suppression threshold for object detection")
    
    # Motion feature options
    parser.add_argument("--use_optical_flow", action="store_true",
                       help="Enable optical flow feature extraction")
    parser.add_argument("--use_motion_history", action="store_true",
                       help="Enable motion history feature extraction")
    
    # Trajectory options
    parser.add_argument("--trajectory_length", type=int, default=30,
                       help="Maximum length of object trajectories")
    parser.add_argument("--max_disappeared", type=int, default=30,
                       help="Maximum number of frames an object can disappear before ending tracking")
    
    # Behavior options
    parser.add_argument("--speed_threshold", type=float, default=5.0,
                       help="Speed threshold for behavior classification")
    parser.add_argument("--interaction_threshold", type=float, default=100.0,
                       help="Distance threshold for interaction detection")
    
    # Visual language model options
    parser.add_argument("--enable_vl", action="store_true",
                       help="Enable visual language model")
    parser.add_argument("--model_version", type=str, default="qwen-vl-chat",
                       help="Visual language model version")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run the visual language model on (cpu or cuda)")
    parser.add_argument("--vl_interval", type=int, default=30,
                       help="Interval in frames for visual language processing")
    parser.add_argument("--num_entities", type=int, default=10,
                       help="Maximum number of entities in the scene graph")
    
    # Alert options
    parser.add_argument("--rules_config", type=str, default="config/rules.json",
                       help="Path to rules configuration file")
    parser.add_argument("--notification_config", type=str, default="config/notification.json",
                       help="Path to notification configuration file")
    parser.add_argument("--alert_interval", type=float, default=10.0,
                       help="Minimum interval between consecutive alerts (seconds)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Directory to save output results")
    parser.add_argument("--save_frames", action="store_true",
                       help="Save processed frames")
    parser.add_argument("--save_interval", type=int, default=30,
                       help="Interval in frames for saving output")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    system = VideoAnalysisSystem(args)
    success = system.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 