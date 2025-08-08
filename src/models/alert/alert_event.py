import time
import uuid
import json
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from models.alert_rule import AlertLevel


@dataclass
class AlertEvent:
    """
    Alert event data structure that contains all information about an alert.
    """
    id: str                      # Unique event identifier
    rule_id: str                 # ID of the rule that triggered this alert
    level: AlertLevel            # Alert level
    source_type: str             # Type of source that triggered the alert
    timestamp: float             # Unix timestamp when the alert was generated
    message: str                 # Human-readable message
    details: Dict[str, Any]      # Detailed information about the alert
    frame_idx: int               # Frame index when the alert was triggered
    frame: Optional[np.ndarray] = None  # Frame image when the alert was triggered
    thumbnail: Optional[np.ndarray] = None  # Small thumbnail of the frame
    
    # Additional metadata
    acknowledged: bool = False   # Whether the alert has been acknowledged
    related_events: List[str] = field(default_factory=list)  # IDs of related events
    
    @classmethod
    def create(cls, rule_id: str, level: AlertLevel, source_type: str, 
               message: str, details: Dict[str, Any], frame_idx: int,
               frame: Optional[np.ndarray] = None) -> 'AlertEvent':
        """
        Create a new alert event.
        
        Args:
            rule_id: ID of the rule that triggered this alert
            level: Alert level
            source_type: Type of source that triggered the alert
            message: Human-readable message
            details: Detailed information about the alert
            frame_idx: Frame index when the alert was triggered
            frame: Frame image when the alert was triggered
            
        Returns:
            New AlertEvent instance
        """
        event_id = f"evt_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        timestamp = time.time()
        
        # Create thumbnail if frame is provided
        thumbnail = None
        if frame is not None:
            # Resize to small thumbnail for storage efficiency
            h, w = frame.shape[:2]
            thumbnail_size = (min(320, w), min(240, h * 320 // w))
            thumbnail = cv2.resize(frame, thumbnail_size)
        
        return cls(
            id=event_id,
            rule_id=rule_id,
            level=level,
            source_type=source_type,
            timestamp=timestamp,
            message=message,
            details=details,
            frame_idx=frame_idx,
            frame=frame.copy() if frame is not None else None,
            thumbnail=thumbnail
        )
    
    def to_dict(self, include_images: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Args:
            include_images: Whether to include encoded images
            
        Returns:
            Dictionary representation of the event
        """
        result = {
            'id': self.id,
            'rule_id': self.rule_id,
            'level': self.level.name,
            'source_type': self.source_type,
            'timestamp': self.timestamp,
            'message': self.message,
            'details': self.details,
            'frame_idx': self.frame_idx,
            'acknowledged': self.acknowledged,
            'related_events': self.related_events,
            'datetime': datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if include_images and self.thumbnail is not None:
            # Encode thumbnail as base64 if requested
            import base64
            _, buffer = cv2.imencode('.jpg', self.thumbnail)
            result['thumbnail_base64'] = base64.b64encode(buffer).decode('utf-8')
        
        return result
    
    def save_images(self, output_dir: str) -> Dict[str, str]:
        """
        Save images associated with the event to disk.
        
        Args:
            output_dir: Directory to save images
            
        Returns:
            Dictionary with paths to saved images
        """
        import os
        paths = {}
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Timestamp for unique filenames
        ts = datetime.fromtimestamp(self.timestamp).strftime('%Y%m%d_%H%M%S')
        
        # Save full frame if available
        if self.frame is not None:
            frame_path = os.path.join(output_dir, f"{self.id}_{ts}_frame.jpg")
            cv2.imwrite(frame_path, self.frame)
            paths['frame'] = frame_path
        
        # Save thumbnail if available
        if self.thumbnail is not None:
            thumb_path = os.path.join(output_dir, f"{self.id}_{ts}_thumb.jpg")
            cv2.imwrite(thumb_path, self.thumbnail)
            paths['thumbnail'] = thumb_path
        
        return paths


class AlertEventStore:
    """
    Store for alert events with persistence capabilities.
    """
    
    def __init__(self, max_events: int = 1000, output_dir: str = "alert_events"):
        """
        Initialize the alert event store.
        
        Args:
            max_events: Maximum number of events to keep in memory
            output_dir: Directory to save event images
        """
        self.events = []
        self.max_events = max_events
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def add_event(self, event: AlertEvent, save_images: bool = True) -> str:
        """
        Add a new event to the store.
        
        Args:
            event: Alert event to add
            save_images: Whether to save event images to disk
            
        Returns:
            ID of the added event
        """
        # Save images if requested
        if save_images:
            event.save_images(self.output_dir)
        
        # Add event to store
        self.events.append(event)
        
        # Trim if needed
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        return event.id
    
    def get_event(self, event_id: str) -> Optional[AlertEvent]:
        """
        Get an event by ID.
        
        Args:
            event_id: ID of the event to get
            
        Returns:
            Event if found, None otherwise
        """
        for event in self.events:
            if event.id == event_id:
                return event
        return None
    
    def get_events(self, count: int = None, level: Optional[AlertLevel] = None, 
                  source_type: Optional[str] = None, 
                  acknowledged: Optional[bool] = None) -> List[AlertEvent]:
        """
        Get filtered events.
        
        Args:
            count: Maximum number of events to return (newest first)
            level: Filter by alert level
            source_type: Filter by source type
            acknowledged: Filter by acknowledged status
            
        Returns:
            List of matching events
        """
        filtered = self.events
        
        # Apply filters
        if level is not None:
            filtered = [e for e in filtered if e.level == level]
        
        if source_type is not None:
            filtered = [e for e in filtered if e.source_type == source_type]
        
        if acknowledged is not None:
            filtered = [e for e in filtered if e.acknowledged == acknowledged]
        
        # Sort by timestamp (newest first)
        filtered.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Limit count if specified
        if count is not None:
            filtered = filtered[:count]
        
        return filtered
    
    def acknowledge_event(self, event_id: str) -> bool:
        """
        Mark an event as acknowledged.
        
        Args:
            event_id: ID of the event to acknowledge
            
        Returns:
            True if acknowledged, False if not found
        """
        event = self.get_event(event_id)
        if event:
            event.acknowledged = True
            return True
        return False
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save events to a JSON file.
        
        Args:
            filepath: Path to save the events
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert events to dictionaries without images
            events_data = [event.to_dict(include_images=False) for event in self.events]
            
            with open(filepath, 'w') as f:
                json.dump(events_data, f, indent=2)
            
            return True
        except Exception as e:
            import logging
            logging.error(f"Failed to save events to {filepath}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored events.
        
        Returns:
            Dictionary with event statistics
        """
        if not self.events:
            return {
                'total': 0,
                'by_level': {},
                'by_source': {},
                'acknowledged': 0
            }
        
        # Count events by level and source
        levels = {}
        sources = {}
        acknowledged = 0
        
        for event in self.events:
            # Count by level
            level_name = event.level.name
            levels[level_name] = levels.get(level_name, 0) + 1
            
            # Count by source
            sources[event.source_type] = sources.get(event.source_type, 0) + 1
            
            # Count acknowledged
            if event.acknowledged:
                acknowledged += 1
        
        return {
            'total': len(self.events),
            'by_level': levels,
            'by_source': sources,
            'acknowledged': acknowledged
        } 