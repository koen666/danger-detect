import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from models.motion.motion_features import MotionFeature


class ObjectTrajectory:
    """Class for storing and analyzing object trajectories."""
    
    def __init__(self, max_history: int = 60):
        """
        Initialize the object trajectory.
        
        Args:
            max_history: Maximum number of positions to store
        """
        self.positions = []  # List of (x, y, frame_idx)
        self.velocities = []  # List of (vx, vy, frame_idx)
        self.boxes = []  # List of (x1, y1, x2, y2, frame_idx)
        self.class_names = []  # List of class names
        self.motion_features = []  # List of motion features
        self.max_history = max_history
        self.last_update_time = 0
        
        # Statistical features
        self.avg_speed = 0
        self.max_speed = 0
        self.direction_changes = 0
        self.area_covered = 0
        
        # Analysis results
        self.behavior_history = []  # List of behavior analysis results
    
    def update(self, track: Dict[str, Any], motion_features: List[MotionFeature], frame_idx: int):
        """
        Update the trajectory with new track data.
        
        Args:
            track: Track dictionary from object tracker
            motion_features: Motion features associated with this object
            frame_idx: Current frame index
        """
        # Extract data from track
        box = track['box']
        class_name = track['class_name']
        
        # Add box and class name
        self.boxes.append((box[0], box[1], box[2], box[3], frame_idx))
        self.class_names.append(class_name)
        
        # Calculate center position
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        
        # Add position
        self.positions.append((cx, cy, frame_idx))
        
        # Calculate velocity if we have previous positions
        if len(self.positions) > 1:
            prev_x, prev_y, prev_frame = self.positions[-2]
            
            # Calculate time difference in frames
            frame_diff = frame_idx - prev_frame
            
            if frame_diff > 0:
                # Calculate velocity components
                vx = (cx - prev_x) / frame_diff
                vy = (cy - prev_y) / frame_diff
                
                # Add velocity
                self.velocities.append((vx, vy, frame_idx))
                
                # Update statistics
                speed = np.sqrt(vx*vx + vy*vy)
                
                # Update max speed
                self.max_speed = max(self.max_speed, speed)
                
                # Update average speed
                self.avg_speed = (self.avg_speed * (len(self.velocities) - 1) + speed) / len(self.velocities)
                
                # Check for direction change
                if len(self.velocities) > 1:
                    prev_vx, prev_vy, _ = self.velocities[-2]
                    
                    # Calculate dot product to check direction change
                    dot = prev_vx * vx + prev_vy * vy
                    mag1 = np.sqrt(prev_vx*prev_vx + prev_vy*prev_vy)
                    mag2 = np.sqrt(vx*vx + vy*vy)
                    
                    if mag1 > 0 and mag2 > 0:
                        cos_angle = dot / (mag1 * mag2)
                        cos_angle = max(-1, min(1, cos_angle))  # Ensure value is in [-1, 1]
                        angle = np.arccos(cos_angle)
                        
                        # Count direction changes greater than 45 degrees
                        if angle > np.pi/4:
                            self.direction_changes += 1
        
        # Add motion features
        for feature in motion_features:
            if feature.object_id == track['id']:
                self.motion_features.append(feature)
        
        # Update last update time
        self.last_update_time = time.time()
        
        # Trim history if needed
        if len(self.positions) > self.max_history:
            self.positions.pop(0)
        if len(self.velocities) > self.max_history:
            self.velocities.pop(0)
        if len(self.boxes) > self.max_history:
            self.boxes.pop(0)
        if len(self.class_names) > self.max_history:
            self.class_names.pop(0)
        if len(self.motion_features) > self.max_history:
            self.motion_features.pop(0)
        
        # Update area covered
        if len(self.positions) > 1:
            x_values = [p[0] for p in self.positions]
            y_values = [p[1] for p in self.positions]
            
            width = max(x_values) - min(x_values)
            height = max(y_values) - min(y_values)
            
            self.area_covered = width * height
    
    def get_features(self) -> Dict[str, Any]:
        """
        Get statistical features for behavior analysis.
        
        Returns:
            Dictionary of features
        """
        features = {
            'avg_speed': self.avg_speed,
            'max_speed': self.max_speed,
            'direction_changes': self.direction_changes,
            'area_covered': self.area_covered,
            'trajectory_length': len(self.positions),
            'class_name': self.class_names[-1] if self.class_names else 'unknown'
        }
        
        # Add velocity features if available
        if self.velocities:
            vx_values = [v[0] for v in self.velocities]
            vy_values = [v[1] for v in self.velocities]
            
            # Calculate average velocity components
            features['avg_vx'] = np.mean(vx_values)
            features['avg_vy'] = np.mean(vy_values)
            
            # Calculate velocity variance
            features['var_vx'] = np.var(vx_values)
            features['var_vy'] = np.var(vy_values)
            
            # Calculate acceleration
            if len(self.velocities) > 1:
                acc_x = np.diff([v[0] for v in self.velocities])
                acc_y = np.diff([v[1] for v in self.velocities])
                
                features['max_acc'] = max(np.sqrt(acc_x**2 + acc_y**2))
                features['avg_acc'] = np.mean(np.sqrt(acc_x**2 + acc_y**2))
        
        return features
    
    def add_behavior_result(self, result: Any):
        """
        Add a behavior analysis result.
        
        Args:
            result: Behavior analysis result
        """
        self.behavior_history.append(result)
        
        # Trim history if needed
        if len(self.behavior_history) > self.max_history:
            self.behavior_history.pop(0) 
