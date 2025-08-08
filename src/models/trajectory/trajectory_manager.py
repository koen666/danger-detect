import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from models.motion.motion_features import MotionFeature
from models.trajectory.trajectory import ObjectTrajectory
from models.trajectory.interaction_detector import InteractionDetector


class TrajectoryManager:
    """Manager class for tracking and analyzing object trajectories."""
    
    def __init__(self, max_trajectory_length=60, max_disappeared=30, 
                 interaction_distance=100, interaction_threshold=3):
        """
        Initialize the trajectory manager.
        
        Args:
            max_trajectory_length: Maximum number of points to keep in trajectory
            max_disappeared: Maximum number of frames an object can disappear before being removed
            interaction_distance: Distance threshold for interaction detection
            interaction_threshold: Number of frames for interaction confirmation
        """
        self.trajectories = {}  # Object ID -> trajectory data
        self.active_object_ids = set()
        self.disappeared_counts = {}  # Object ID -> number of frames disappeared
        
        self.max_trajectory_length = max_trajectory_length
        self.max_disappeared = max_disappeared
        
        # Create interaction detector
        self.interaction_detector = InteractionDetector(
            distance_threshold=interaction_distance,
            count_threshold=interaction_threshold
        )
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TrajectoryManager')
    
    def update(self, tracked_objects):
        """
        Update trajectories with new tracked objects.
        
        Args:
            tracked_objects: List of tracked objects from object tracker
            
        Returns:
            List of active object IDs
        """
        # Get current tracked object IDs
        current_ids = {obj.id for obj in tracked_objects if hasattr(obj, 'id')}
        
        # Clear the active set and build it again
        self.active_object_ids.clear()
        
        # Update disappeared counts
        for obj_id in list(self.disappeared_counts.keys()):
            if obj_id not in current_ids:
                self.disappeared_counts[obj_id] += 1
                
                # Remove if disappeared for too long
                if self.disappeared_counts[obj_id] > self.max_disappeared:
                    del self.disappeared_counts[obj_id]
                    if obj_id in self.trajectories:
                        del self.trajectories[obj_id]
            else:
                self.disappeared_counts[obj_id] = 0
                self.active_object_ids.add(obj_id)
        
        # Add new objects to disappearance tracker
        for obj_id in current_ids:
            if obj_id not in self.disappeared_counts:
                self.disappeared_counts[obj_id] = 0
                self.active_object_ids.add(obj_id)
        
        # Update trajectories
        for obj in tracked_objects:
            if not hasattr(obj, 'id'):
                continue
                
            obj_id = obj.id
            
            # Get center position
            if hasattr(obj, 'x1') and hasattr(obj, 'y1') and hasattr(obj, 'x2') and hasattr(obj, 'y2'):
                center_x = (obj.x1 + obj.x2) / 2
                center_y = (obj.y1 + obj.y2) / 2
                position = (center_x, center_y)
            else:
                # Skip if we can't determine position
                continue
            
            # Initialize trajectory if new
            if obj_id not in self.trajectories:
                self.trajectories[obj_id] = {
                    'positions': [position],
                    'timestamps': [time.time()],
                    'class_name': getattr(obj, 'class_name', None),
                    'confidence': getattr(obj, 'confidence', 0),
                    'speeds': [],
                    'direction_changes': 0,
                    'is_stationary': False,
                    'area_covered': 0
                }
            else:
                # Add to existing trajectory
                self.trajectories[obj_id]['positions'].append(position)
                self.trajectories[obj_id]['timestamps'].append(time.time())
                
                # Update class name and confidence if available
                if hasattr(obj, 'class_name'):
                    self.trajectories[obj_id]['class_name'] = obj.class_name
                if hasattr(obj, 'confidence'):
                    self.trajectories[obj_id]['confidence'] = obj.confidence
                
                # Limit trajectory length
                if len(self.trajectories[obj_id]['positions']) > self.max_trajectory_length:
                    self.trajectories[obj_id]['positions'] = self.trajectories[obj_id]['positions'][-self.max_trajectory_length:]
                    self.trajectories[obj_id]['timestamps'] = self.trajectories[obj_id]['timestamps'][-self.max_trajectory_length:]
                
                # Calculate speed
                if len(self.trajectories[obj_id]['positions']) >= 2:
                    prev_pos = self.trajectories[obj_id]['positions'][-2]
                    curr_pos = self.trajectories[obj_id]['positions'][-1]
                    
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    dt = self.trajectories[obj_id]['timestamps'][-1] - self.trajectories[obj_id]['timestamps'][-2]
                    speed = distance / dt if dt > 0 else 0
                    
                    self.trajectories[obj_id]['speeds'].append(speed)
                    
                    # Limit speed history
                    if len(self.trajectories[obj_id]['speeds']) > self.max_trajectory_length:
                        self.trajectories[obj_id]['speeds'] = self.trajectories[obj_id]['speeds'][-self.max_trajectory_length:]
                    
                    # Calculate if stationary
                    avg_speed = np.mean(self.trajectories[obj_id]['speeds'][-min(10, len(self.trajectories[obj_id]['speeds'])):])
                    self.trajectories[obj_id]['is_stationary'] = avg_speed < 5.0
                
                # Calculate direction changes
                if len(self.trajectories[obj_id]['positions']) >= 3:
                    p1 = self.trajectories[obj_id]['positions'][-3]
                    p2 = self.trajectories[obj_id]['positions'][-2]
                    p3 = self.trajectories[obj_id]['positions'][-1]
                    
                    v1 = (p2[0] - p1[0], p2[1] - p1[1])
                    v2 = (p3[0] - p2[0], p3[1] - p2[1])
                    
                    # Calculate angle between vectors
                    dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                    mag1 = np.sqrt(v1[0]*v1[0] + v1[1]*v1[1])
                    mag2 = np.sqrt(v2[0]*v2[0] + v2[1]*v2[1])
                    
                    if mag1 > 0 and mag2 > 0:
                        cos_angle = dot_product / (mag1 * mag2)
                        # Clamp to [-1, 1] to avoid numerical errors
                        cos_angle = max(-1, min(1, cos_angle))
                        angle = np.arccos(cos_angle) * 180 / np.pi
                        
                        # Detect significant direction change
                        if angle > 45:
                            self.trajectories[obj_id]['direction_changes'] += 1
                
                # Calculate area covered
                if len(self.trajectories[obj_id]['positions']) >= 2:
                    points = np.array(self.trajectories[obj_id]['positions'])
                    x_min, y_min = np.min(points, axis=0)
                    x_max, y_max = np.max(points, axis=0)
                    
                    self.trajectories[obj_id]['area_covered'] = (x_max - x_min) * (y_max - y_min)
        
        # Update interaction detector
        self.interaction_detector.update(self.get_trajectory_positions())
        
        return list(self.active_object_ids)
    
    def get_trajectory_positions(self):
        """
        Get current positions of all active trajectories.
        
        Returns:
            Dictionary of object ID to current position
        """
        positions = {}
        for obj_id in self.active_object_ids:
            if obj_id in self.trajectories and self.trajectories[obj_id]['positions']:
                positions[obj_id] = self.trajectories[obj_id]['positions'][-1]
        return positions
    
    def get_active_trajectories(self):
        """
        Get all active trajectories.
        
        Returns:
            List of active trajectories
        """
        active_trajectories = []
        for obj_id in self.active_object_ids:
            if obj_id in self.trajectories:
                traj = self.trajectories[obj_id].copy()
                traj['id'] = obj_id
                traj['position'] = traj['positions'][-1] if traj['positions'] else None
                active_trajectories.append(traj)
        return active_trajectories
    
    def get_interaction_detector(self):
        """
        Get the interaction detector instance.
        
        Returns:
            InteractionDetector: The interaction detector
        """
        return self.interaction_detector
    
    def get_trajectory(self, object_id):
        """
        Get trajectory for a specific object.
        
        Args:
            object_id: Object ID
            
        Returns:
            Object trajectory or None if not found
        """
        if object_id in self.trajectories:
            traj = self.trajectories[object_id].copy()
            traj['id'] = object_id
            traj['position'] = traj['positions'][-1] if traj['positions'] else None
            return traj
        return None
    
    def reset(self):
        """Reset all trajectories and counts."""
        self.trajectories = {}
        self.active_object_ids.clear()
        self.disappeared_counts = {}
        self.interaction_detector.reset() 