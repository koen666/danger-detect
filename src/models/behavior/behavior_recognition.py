import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

from models.motion.motion_features import MotionFeature

from models.trajectory.trajectory_manager import TrajectoryManager

from models.behavior.behavior_types import BehaviorType, Behavior, Interaction


class BehaviorRecognizer:
    """
    Recognizes behaviors and interactions based on trajectory and motion features.
    """
    
    def __init__(self, speed_threshold=10.0, interaction_threshold=50.0):
        """
        Initialize the behavior recognizer.
        
        Args:
            speed_threshold: Threshold for speed-based behaviors
            interaction_threshold: Threshold for interaction detection
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('BehaviorRecognizer')
        
        self.speed_threshold = speed_threshold
        self.interaction_threshold = interaction_threshold
        
        self.is_initialized = True
        self.logger.info("Behavior recognizer initialized")
    
    def analyze(self, trajectories, motion_features, interaction_detector):
        """
        Analyze trajectories and motion features to identify behaviors and interactions.
        
        Args:
            trajectories: List of trajectories from trajectory manager
            motion_features: List of motion features from motion manager
            interaction_detector: Interaction detector instance
            
        Returns:
            Tuple of (behaviors, interactions)
        """
        if not trajectories:
            return [], []
        
        # Analyze individual behaviors
        behaviors = self.analyze_trajectories(trajectories, motion_features)
        
        # Detect interactions between objects
        interactions = self.detect_interactions(trajectories, interaction_detector)
        
        return behaviors, interactions
    
    def analyze_trajectories(self, trajectories, motion_features):
        """
        Analyze individual trajectories to identify behaviors.
        
        Args:
            trajectories: List of trajectories
            motion_features: List of motion features
            
        Returns:
            List of behavior results
        """
        behaviors = []
        
        for trajectory in trajectories:
            # Skip if the trajectory doesn't have enough data
            if 'speeds' not in trajectory or not trajectory['speeds']:
                continue
            
            obj_id = trajectory['id']
            
            # Get latest position
            position = trajectory['position']
            
            # Calculate average speed
            avg_speed = np.mean(trajectory['speeds'][-min(10, len(trajectory['speeds'])):])
            
            # Calculate motion features for this object
            obj_motion_features = []
            for feature in motion_features:
                if hasattr(feature, 'object_id') and feature.object_id == obj_id:
                    obj_motion_features.append(feature)
            
            # Determine behavior based on speed and trajectory features
            behavior_type = BehaviorType.UNKNOWN
            confidence = 0.5
            
            if trajectory['is_stationary']:
                behavior_type = BehaviorType.STATIC
                confidence = 0.9
            elif avg_speed < self.speed_threshold / 2:
                behavior_type = BehaviorType.WALKING
                confidence = 0.8
            elif avg_speed >= self.speed_threshold:
                behavior_type = BehaviorType.RUNNING
                confidence = 0.8
            
            # Check for loitering behavior
            if trajectory['is_stationary'] and trajectory['area_covered'] < 2000:
                behavior_type = BehaviorType.LOITERING
                confidence = 0.7
            
            # Check for wandering behavior
            if trajectory['direction_changes'] > 3 and avg_speed < self.speed_threshold:
                behavior_type = BehaviorType.WANDERING
                confidence = 0.7
            
            # Create behavior object
            behavior = Behavior(
                object_id=obj_id,
                behavior_type=behavior_type,
                confidence=confidence,
                position=position,
                timestamp=trajectory['timestamps'][-1] if trajectory['timestamps'] else None,
                metadata={
                    'class_name': trajectory.get('class_name', 'unknown'),
                    'speed': avg_speed,
                    'direction_changes': trajectory.get('direction_changes', 0),
                    'is_stationary': trajectory.get('is_stationary', False),
                    'area_covered': trajectory.get('area_covered', 0)
                }
            )
            
            behaviors.append(behavior)
        
        return behaviors
    
    def detect_interactions(self, trajectories, interaction_detector):
        """
        Detect interactions between objects.
        
        Args:
            trajectories: List of trajectories
            interaction_detector: Interaction detector instance
            
        Returns:
            List of interaction results
        """
        interactions = []
        
        # Skip if we don't have enough objects for interactions
        if len(trajectories) < 2:
            return interactions
        
        # Get current interactions from detector
        detector_interactions = interaction_detector.get_interactions()
        
        for interaction_pair, details in detector_interactions.items():
            obj_id1, obj_id2 = interaction_pair
            
            # Get trajectories for involved objects
            traj1 = next((t for t in trajectories if t['id'] == obj_id1), None)
            traj2 = next((t for t in trajectories if t['id'] == obj_id2), None)
            
            if traj1 is None or traj2 is None:
                continue
            
            # Determine interaction type based on the nature of the trajectories
            interaction_type = BehaviorType.GATHERING  # Default
            confidence = 0.6
            
            # Check if both objects are stationary
            if traj1.get('is_stationary', False) and traj2.get('is_stationary', False):
                interaction_type = BehaviorType.GATHERING
                confidence = 0.8
            
            # Check if both objects are moving fast
            elif (np.mean(traj1.get('speeds', [0])[-5:]) > self.speed_threshold and 
                  np.mean(traj2.get('speeds', [0])[-5:]) > self.speed_threshold):
                interaction_type = BehaviorType.FOLLOWING
                confidence = 0.7
            
            # Calculate midpoint between objects
            pos1 = traj1['position']
            pos2 = traj2['position']
            mid_point = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
            
            # Create interaction object
            interaction = Interaction(
                object_ids=[obj_id1, obj_id2],
                behavior_type=interaction_type,
                confidence=confidence,
                position=mid_point,
                timestamp=max(traj1['timestamps'][-1], traj2['timestamps'][-1]) if traj1['timestamps'] and traj2['timestamps'] else None,
                metadata={
                    'distance': details.get('distance', 0),
                    'duration': details.get('count', 0),
                    'class1': traj1.get('class_name', 'unknown'),
                    'class2': traj2.get('class_name', 'unknown')
                }
            )
            
            interactions.append(interaction)
        
        return interactions
    
    def visualize_behaviors(self, frame, behaviors, interactions):
        """
        Visualize behaviors and interactions on a frame.
        
        Args:
            frame: Frame to visualize on
            behaviors: List of behavior results
            interactions: List of interaction results
            
        Returns:
            Visualization frame
        """
        if frame is None:
            return None
        
        vis_frame = frame.copy()
        
        # Define colors for each behavior type
        colors = {
            BehaviorType.STATIC: (200, 200, 200),     # Gray
            BehaviorType.WALKING: (0, 255, 0),        # Green
            BehaviorType.RUNNING: (0, 165, 255),      # Orange
            BehaviorType.FALLING: (0, 0, 255),        # Red
            BehaviorType.FIGHTING: (255, 0, 0),       # Blue
            BehaviorType.GATHERING: (255, 0, 255),    # Purple
            BehaviorType.WANDERING: (255, 255, 0),    # Cyan
            BehaviorType.LOITERING: (128, 0, 255),    # Purple
            BehaviorType.FOLLOWING: (255, 128, 0),    # Sky Blue
            BehaviorType.UNKNOWN: (200, 200, 200)     # Gray
        }
        
        # Draw behaviors
        for behavior in behaviors:
            if not hasattr(behavior, 'position') or not behavior.position:
                continue
                
            x, y = behavior.position
            color = colors.get(behavior.behavior_type, (200, 200, 200))
            
            # Draw behavior type and confidence
            label = f"{behavior.behavior_type.name}: {behavior.confidence:.2f}"
            cv2.putText(vis_frame, label, (int(x), int(y) - 15),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw interactions
        for interaction in interactions:
            if not hasattr(interaction, 'position') or not interaction.position:
                continue
                
            x, y = interaction.position
            color = colors.get(interaction.behavior_type, (200, 200, 200))
            
            # Draw a line between interacting objects
            if hasattr(interaction, 'object_ids') and len(interaction.object_ids) >= 2:
                obj_id1, obj_id2 = interaction.object_ids[:2]
                
                # Draw interaction type
                label = f"{interaction.behavior_type.name}: {interaction.confidence:.2f}"
                cv2.putText(vis_frame, label, (int(x), int(y) - 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw a circle at the midpoint
                cv2.circle(vis_frame, (int(x), int(y)), 5, color, -1)
        
        return vis_frame 
