import numpy as np
import logging
from typing import Dict, List, Tuple, Set, Any


class InteractionDetector:
    """Detects interactions between objects based on their trajectories."""
    
    def __init__(self, distance_threshold: float = 100, count_threshold: int = 2):
        """
        Initialize the interaction detector.
        
        Args:
            distance_threshold: Maximum distance between objects to be considered interacting
            count_threshold: Minimum number of consecutive frames for interaction confirmation
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('InteractionDetector')
        
        self.distance_threshold = distance_threshold
        self.count_threshold = count_threshold
        self.interaction_counts = {}  # (obj_id1, obj_id2) -> count
        self.detected_interactions = {}  # (obj_id1, obj_id2) -> interaction details
    
    def update(self, object_positions):
        """
        Update the interaction detector with new object positions.
        
        Args:
            object_positions: Dictionary of object ID to position (x, y)
            
        Returns:
            Dictionary of detected interactions
        """
        # Check all pairs of objects
        current_interacting_pairs = set()
        
        # Find pairs of close objects
        object_ids = list(object_positions.keys())
        for i, obj_id1 in enumerate(object_ids):
            pos1 = object_positions[obj_id1]
            
            for j in range(i + 1, len(object_ids)):
                obj_id2 = object_ids[j]
                pos2 = object_positions[obj_id2]
                
                # Calculate distance
                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Determine pair key (always order by smaller ID first)
                pair = (min(obj_id1, obj_id2), max(obj_id1, obj_id2))
                
                if distance < self.distance_threshold:
                    current_interacting_pairs.add(pair)
                    
                    # Update count
                    if pair in self.interaction_counts:
                        self.interaction_counts[pair] += 1
                    else:
                        self.interaction_counts[pair] = 1
                    
                    # Check if interaction threshold is reached
                    if self.interaction_counts[pair] >= self.count_threshold:
                        self.detected_interactions[pair] = {
                            'distance': distance,
                            'count': self.interaction_counts[pair],
                            'position1': pos1,
                            'position2': pos2
                        }
                else:
                    # Decrease count if objects are far apart
                    if pair in self.interaction_counts:
                        self.interaction_counts[pair] -= 1
                        if self.interaction_counts[pair] <= 0:
                            self.interaction_counts.pop(pair, None)
                            self.detected_interactions.pop(pair, None)
        
        # Remove interactions for pairs not currently detected
        for pair in list(self.interaction_counts.keys()):
            if pair not in current_interacting_pairs:
                self.interaction_counts[pair] -= 1
                if self.interaction_counts[pair] <= 0:
                    self.interaction_counts.pop(pair, None)
                    self.detected_interactions.pop(pair, None)
        
        return self.detected_interactions
    
    def get_interactions(self):
        """
        Get the current detected interactions.
        
        Returns:
            Dictionary of detected interactions
        """
        return self.detected_interactions
    
    def detect_gatherings(self, object_positions):
        """
        Detect groups of objects that are close to each other.
        
        Args:
            object_positions: Dictionary of object ID to position (x, y)
            
        Returns:
            List of sets of object IDs
        """
        groups = []
        
        # Skip if not enough objects
        if len(object_positions) < self.count_threshold:
            return groups
        
        # Build adjacency graph
        adjacency = {obj_id: set() for obj_id in object_positions}
        
        # Find pairs of close objects
        object_ids = list(object_positions.keys())
        for i, obj_id1 in enumerate(object_ids):
            pos1 = object_positions[obj_id1]
            
            for j in range(i + 1, len(object_ids)):
                obj_id2 = object_ids[j]
                pos2 = object_positions[obj_id2]
                
                # Calculate distance
                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < self.distance_threshold:
                    adjacency[obj_id1].add(obj_id2)
                    adjacency[obj_id2].add(obj_id1)
        
        # Find connected components
        visited = set()
        
        for obj_id in object_positions:
            if obj_id not in visited:
                # Start a new group
                group = set()
                queue = [obj_id]
                
                while queue:
                    current = queue.pop(0)
                    
                    if current not in visited:
                        visited.add(current)
                        group.add(current)
                        
                        # Add neighbors to queue
                        for neighbor in adjacency[current]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                if len(group) >= self.count_threshold:
                    groups.append(group)
        
        return groups
    
    def calculate_group_positions(self, groups, object_positions):
        """
        Calculate the average position of each group.
        
        Args:
            groups: List of sets of object IDs
            object_positions: Dictionary of object ID to position (x, y)
            
        Returns:
            List of average positions (x, y) for each group
        """
        group_positions = []
        
        for group in groups:
            positions = []
            for obj_id in group:
                if obj_id in object_positions:
                    positions.append(object_positions[obj_id])
            
            if positions:
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                group_positions.append((avg_x, avg_y))
            else:
                group_positions.append((0, 0))  # Fallback
        
        return group_positions
    
    def reset(self):
        """Reset the interaction detector."""
        self.interaction_counts = {}
        self.detected_interactions = {} 