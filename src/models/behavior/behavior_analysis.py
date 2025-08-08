import cv2
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import json

from models.motion.motion_features import MotionFeatureManager
from models.behavior.behavior_recognition import BehaviorRecognitionSystem, BehaviorAnalysisResult, BehaviorType
from models.trajectory.trajectory import ObjectTrajectory
from utils.motion_utils import (
    create_motion_heatmap,
    apply_colormap,
    blend_heatmap_with_frame,
    create_motion_field_visualization,
    visualize_trajectory,
    save_motion_features,
    save_behavior_results,
    plot_trajectory_features,
    extract_relevant_frames
)


class BehaviorAnalysisSystem:
    """
    Integrated system for motion feature extraction, behavior recognition, and visualization.
    """
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480,
                 use_optical_flow: bool = True, use_motion_history: bool = True,
                 max_trajectory_history: int = 60):
        """
        Initialize the behavior analysis system.
        
        Args:
            frame_width: Width of the frame
            frame_height: Height of the frame
            use_optical_flow: Whether to use optical flow extractor
            use_motion_history: Whether to use motion history extractor
            max_trajectory_history: Maximum number of frames to keep in trajectory history
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('BehaviorAnalysisSystem')
        
        # Initialize frame dimensions
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Initialize motion feature manager
        self.motion_feature_manager = MotionFeatureManager(
            use_optical_flow=use_optical_flow,
            use_motion_history=use_motion_history
        )
        
        # Initialize behavior recognition system
        self.behavior_recognition = BehaviorRecognitionSystem(
            frame_width=frame_width,
            frame_height=frame_height,
            max_trajectory_history=max_trajectory_history
        )
        
        # Initialize history
        self.motion_features_history = []
        self.behavior_results_history = []
        
        # Maximum history length
        self.max_history_length = 300
        
        # Statistics
        self.frame_count = 0
        self.processing_times = []
        
        self.is_initialized = (
            self.motion_feature_manager.is_initialized and 
            self.behavior_recognition.is_initialized
        )
        
        if self.is_initialized:
            self.logger.info("Behavior analysis system initialized")
    
    def process_frame(self, frame: np.ndarray, tracks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a frame to extract motion features and analyze behavior.
        
        Args:
            frame: Input frame
            tracks: Object tracks from object tracker (optional)
            
        Returns:
            Dictionary of results:
            - 'motion_features': List of motion features
            - 'behavior_results': List of behavior analysis results
            - 'processing_time': Processing time in seconds
        """
        if not self.is_initialized or frame is None:
            return {'motion_features': [], 'behavior_results': [], 'processing_time': 0}
        
        start_time = cv2.getTickCount()
        
        # Step 1: Extract motion features
        motion_features = self.motion_feature_manager.extract_features(frame, tracks)
        
        # Step 2: Analyze behavior
        behavior_results = self.behavior_recognition.update(tracks or [], motion_features)
        
        # Calculate processing time
        processing_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        
        # Update history
        self.motion_features_history.append(motion_features)
        self.behavior_results_history.append(behavior_results)
        
        # Trim history if needed
        if len(self.motion_features_history) > self.max_history_length:
            self.motion_features_history.pop(0)
        if len(self.behavior_results_history) > self.max_history_length:
            self.behavior_results_history.pop(0)
        
        # Update statistics
        self.frame_count += 1
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        return {
            'motion_features': motion_features,
            'behavior_results': behavior_results,
            'processing_time': processing_time
        }
    
    def reset(self):
        """Reset the behavior analysis system."""
        self.motion_feature_manager.reset()
        self.behavior_recognition.reset()
        self.motion_features_history = []
        self.behavior_results_history = []
        self.frame_count = 0
        self.processing_times = []
    
    def visualize_motion_features(self, frame: np.ndarray, motion_features: List[Any] = None) -> np.ndarray:
        """
        Visualize motion features on a frame.
        
        Args:
            frame: Frame to draw on
            motion_features: List of motion features (if None, use latest)
            
        Returns:
            Frame with visualized motion features
        """
        if frame is None:
            return frame
        
        # Use latest features if not provided
        if motion_features is None:
            if not self.motion_features_history:
                return frame
            motion_features = self.motion_features_history[-1]
        
        return self.motion_feature_manager.visualize_features(frame, motion_features)
    
    def visualize_behaviors(self, frame: np.ndarray, behavior_results: List[BehaviorAnalysisResult] = None) -> np.ndarray:
        """
        Visualize behavior analysis results on a frame.
        
        Args:
            frame: Frame to draw on
            behavior_results: Behavior analysis results (if None, use latest)
            
        Returns:
            Frame with visualized behaviors
        """
        if frame is None:
            return frame
        
        # Use latest results if not provided
        if behavior_results is None:
            if not self.behavior_results_history:
                return frame
            behavior_results = self.behavior_results_history[-1]
        
        return self.behavior_recognition.visualize_behaviors(frame, behavior_results)
    
    def create_motion_heatmap(self, frame_shape: Tuple[int, int] = None) -> np.ndarray:
        """
        Create a motion heatmap from recent motion features.
        
        Args:
            frame_shape: Shape of the frame as (height, width), if None, use initialized dimensions
            
        Returns:
            Colored heatmap as BGR image
        """
        if not self.motion_features_history:
            return None
        
        # Get all motion features from recent history (last 30 frames)
        all_features = []
        for features in self.motion_features_history[-30:]:
            all_features.extend(features)
        
        # Use initialized dimensions if not provided
        if frame_shape is None:
            frame_shape = (self.frame_height, self.frame_width)
        
        # Create heatmap
        heatmap = create_motion_heatmap(all_features, frame_shape)
        colored_heatmap = apply_colormap(heatmap)
        
        return colored_heatmap
    
    def visualize_combined_results(self, frame: np.ndarray, include_motion: bool = True, 
                                   include_behavior: bool = True, include_heatmap: bool = False,
                                   heatmap_alpha: float = 0.3) -> np.ndarray:
        """
        Create a combined visualization of motion features and behavior analysis.
        
        Args:
            frame: Input frame
            include_motion: Whether to include motion feature visualization
            include_behavior: Whether to include behavior visualization
            include_heatmap: Whether to include motion heatmap
            heatmap_alpha: Alpha value for heatmap blending
            
        Returns:
            Frame with combined visualizations
        """
        if frame is None:
            return frame
        
        result = frame.copy()
        
        # Add motion features
        if include_motion and self.motion_features_history:
            result = self.visualize_motion_features(result)
        
        # Add motion heatmap
        if include_heatmap:
            heatmap = self.create_motion_heatmap(frame.shape[:2])
            if heatmap is not None:
                result = blend_heatmap_with_frame(result, heatmap, heatmap_alpha)
        
        # Add behavior visualization
        if include_behavior and self.behavior_results_history:
            result = self.visualize_behaviors(result)
        
        # Add processing statistics
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            cv2.putText(
                result,
                f"FPS: {fps:.1f} | Frame: {self.frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        return result
    
    def get_trajectory(self, object_id: int) -> Optional[ObjectTrajectory]:
        """
        Get trajectory for a specific object.
        
        Args:
            object_id: Object ID
            
        Returns:
            Object trajectory or None if not found
        """
        return self.behavior_recognition.get_trajectory(object_id)
    
    def visualize_trajectory(self, frame: np.ndarray, object_id: int) -> np.ndarray:
        """
        Visualize trajectory for a specific object.
        
        Args:
            frame: Frame to draw on
            object_id: Object ID
            
        Returns:
            Frame with visualized trajectory
        """
        trajectory = self.get_trajectory(object_id)
        
        if frame is None or trajectory is None or not trajectory.positions:
            return frame
        
        return visualize_trajectory(frame, trajectory.positions)
    
    def save_analysis_results(self, output_dir: str, prefix: str = "analysis"):
        """
        Save motion features and behavior analysis results to files.
        
        Args:
            output_dir: Output directory
            prefix: Prefix for output files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save motion features
        if self.motion_features_history:
            all_features = []
            for features in self.motion_features_history:
                all_features.extend(features)
            
            motion_filename = os.path.join(output_dir, f"{prefix}_motion_features.json")
            save_motion_features(all_features, motion_filename)
        
        # Save behavior results
        if self.behavior_results_history:
            all_results = []
            for results in self.behavior_results_history:
                all_results.extend(results)
            
            behavior_filename = os.path.join(output_dir, f"{prefix}_behavior_results.json")
            save_behavior_results(all_results, behavior_filename)
        
        # Save statistics
        stats = {
            'frame_count': self.frame_count,
            'avg_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
            'fps': 1.0 / (sum(self.processing_times) / len(self.processing_times)) if self.processing_times else 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        stats_filename = os.path.join(output_dir, f"{prefix}_statistics.json")
        with open(stats_filename, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def generate_behavior_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of detected behaviors.
        
        Returns:
            Dictionary with behavior statistics
        """
        if not self.behavior_results_history:
            return {'behaviors': {}, 'total_count': 0}
        
        # Flatten all behavior results
        all_results = []
        for results in self.behavior_results_history:
            all_results.extend(results)
        
        # Count behaviors by type
        behavior_counts = {}
        for behavior_type in BehaviorType:
            behavior_counts[behavior_type.name] = 0
        
        for result in all_results:
            behavior_counts[result.behavior_type.name] += 1
        
        # Calculate additional statistics
        total_count = len(all_results)
        
        # Sort by frequency (descending)
        sorted_behaviors = sorted(
            [(name, count) for name, count in behavior_counts.items() if count > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'behaviors': {name: {'count': count, 'percentage': (count / total_count) * 100 if total_count > 0 else 0}
                         for name, count in sorted_behaviors},
            'total_count': total_count
        } 
