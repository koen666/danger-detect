import os
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

from models.alert_rule import AlertRule, AlertRuleConfig, AlertLevel
from models.alert_event import AlertEvent, AlertEventStore
from models.alert_processor import AlertProcessor
from models.alert_plugins import NotificationManager


class AlertSystem:
    """
    Main alert system that integrates all alert components and coordinates their operation.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the alert system.
        
        Args:
            config_path: Path to configuration file
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("AlertSystem")
        
        # Initialize configuration
        self.config = self._load_config(config_path)
        
        # Create output directories
        self.output_dir = self.config.get("output_dir", "alert_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize rule configuration
        rules_file = self.config.get("rules_file", os.path.join(self.output_dir, "alert_rules.json"))
        self.rules_config = self._init_rules(rules_file)
        
        # Initialize alert processor
        self.alert_processor = AlertProcessor(self.rules_config)
        
        # Initialize event store
        event_store_config = self.config.get("event_store", {})
        max_events = event_store_config.get("max_events", 1000)
        events_dir = os.path.join(self.output_dir, "events")
        self.event_store = AlertEventStore(max_events=max_events, output_dir=events_dir)
        
        # Initialize notification manager
        notification_config = self.config.get("notifications", {})
        self.notification_manager = NotificationManager(notification_config)
        
        # Initialize state
        self.is_initialized = True
        self.is_running = False
        self.frame_queue = []
        self.queue_lock = threading.Lock()
        self.processing_thread = None
        
        self.logger.info("Alert system initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_path}: {e}")
        
        # Default configuration
        return {
            "output_dir": "alert_output",
            "rules_file": "alert_rules.json",
            "event_store": {
                "max_events": 1000
            },
            "notifications": {
                "file": {
                    "type": "file",
                    "enabled": True,
                    "min_level": "info",
                    "output_file": "alerts.log",
                    "append": True
                }
            },
            "processing": {
                "max_queue_size": 30,
                "process_every_n_frames": 5
            }
        }
    
    def _init_rules(self, rules_file: str) -> AlertRuleConfig:
        """Initialize alert rules from file or create defaults."""
        rules_config = None
        
        # Try to load from file first
        if os.path.exists(rules_file):
            rules_config = AlertRuleConfig.load_from_file(rules_file)
        
        # Create default rules if none were loaded
        if not rules_config or not rules_config.rules:
            rules_config = AlertRuleConfig()
            rules_config.create_default_rules()
            
            # Save the default rules
            rules_config.save_to_file(rules_file)
        
        self.logger.info(f"Loaded {len(rules_config.rules)} alert rules")
        return rules_config
    
    def start(self) -> bool:
        """Start the alert system processing thread."""
        if not self.is_initialized:
            self.logger.error("Alert system not initialized")
            return False
        
        if self.is_running:
            self.logger.warning("Alert system already running")
            return True
        
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Alert system started")
        return True
    
    def stop(self) -> bool:
        """Stop the alert system processing thread."""
        if not self.is_running:
            return True
        
        # Signal thread to stop
        self.is_running = False
        
        # Wait for thread to terminate
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Shutdown notification manager
        self.notification_manager.shutdown()
        
        # Save rules
        rules_file = self.config.get("rules_file", os.path.join(self.output_dir, "alert_rules.json"))
        self.rules_config.save_to_file(rules_file)
        
        # Save events
        events_file = os.path.join(self.output_dir, "alert_events.json")
        self.event_store.save_to_file(events_file)
        
        self.logger.info("Alert system stopped")
        return True
    
    def process_frame(self, frame_idx: int, frame: Any,
                     behavior_results: List[Any] = None,
                     tracks: List[Dict[str, Any]] = None,
                     motion_features: List[Any] = None,
                     scene_data: Dict[str, Any] = None,
                     process_now: bool = False) -> Optional[List[AlertEvent]]:
        """
        Process a frame and associated data for alerts.
        
        Args:
            frame_idx: Frame index
            frame: Current frame
            behavior_results: Behavior analysis results
            tracks: Object tracking results
            motion_features: Motion features
            scene_data: Scene analysis data
            process_now: Process immediately instead of queueing
            
        Returns:
            List of generated alerts if process_now is True, None otherwise
        """
        if not self.is_initialized:
            self.logger.error("Alert system not initialized")
            return None
        
        # Create frame data package
        frame_data = {
            "frame_idx": frame_idx,
            "frame": frame.copy() if frame is not None else None,
            "behavior_results": behavior_results,
            "tracks": tracks,
            "motion_features": motion_features,
            "scene_data": scene_data,
            "timestamp": time.time()
        }
        
        # Process immediately if requested
        if process_now:
            return self._process_frame_data(frame_data)
        
        # Otherwise queue for background processing
        with self.queue_lock:
            # Add to queue
            self.frame_queue.append(frame_data)
            
            # Trim queue if too large
            max_queue_size = self.config.get("processing", {}).get("max_queue_size", 30)
            if len(self.frame_queue) > max_queue_size:
                # Keep most recent frames
                self.frame_queue = self.frame_queue[-max_queue_size:]
        
        return None
    
    def _process_frame_data(self, frame_data: Dict[str, Any]) -> List[AlertEvent]:
        """Process a frame data package and generate alerts."""
        try:
            # Extract data from package
            frame_idx = frame_data["frame_idx"]
            frame = frame_data["frame"]
            behavior_results = frame_data["behavior_results"]
            tracks = frame_data["tracks"]
            motion_features = frame_data["motion_features"]
            scene_data = frame_data["scene_data"]
            
            # Process with alert processor
            alerts = self.alert_processor.process_frame_data(
                frame_idx=frame_idx,
                frame=frame,
                behavior_results=behavior_results,
                tracks=tracks,
                motion_features=motion_features,
                scene_data=scene_data
            )
            
            # Store and notify for each alert
            for alert in alerts:
                # Store in event store
                self.event_store.add_event(alert)
                
                # Send notifications
                self.notification_manager.notify(alert)
                
                # Log alert
                self.logger.info(f"Alert generated: {alert.level.name} - {alert.message}")
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error processing frame data: {e}")
            return []
    
    def _processing_worker(self):
        """Background worker that processes the frame queue."""
        process_every_n = self.config.get("processing", {}).get("process_every_n_frames", 5)
        frame_counter = 0
        
        while self.is_running:
            frame_data_to_process = None
            
            # Get a frame from the queue
            with self.queue_lock:
                if self.frame_queue:
                    frame_counter += 1
                    
                    # Only process every Nth frame to reduce load
                    if frame_counter >= process_every_n:
                        frame_data_to_process = self.frame_queue.pop(0)
                        frame_counter = 0
            
            # Process the frame if one was dequeued
            if frame_data_to_process:
                self._process_frame_data(frame_data_to_process)
            else:
                # Sleep if queue is empty
                time.sleep(0.1)
    
    def get_recent_alerts(self, count: int = 10, 
                         level: Optional[AlertLevel] = None) -> List[Dict[str, Any]]:
        """
        Get recent alerts as dictionaries.
        
        Args:
            count: Maximum number of alerts to return
            level: Filter by alert level
            
        Returns:
            List of alert dictionaries
        """
        events = self.event_store.get_events(count=count, level=level)
        return [event.to_dict() for event in events]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """
        Get statistics about alerts.
        
        Returns:
            Dictionary with alert statistics
        """
        return self.event_store.get_stats()
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            True if acknowledged, False if not found
        """
        return self.event_store.acknowledge_event(alert_id)
    
    def set_rule_enabled(self, rule_id: str, enabled: bool) -> bool:
        """
        Enable or disable an alert rule.
        
        Args:
            rule_id: ID of the rule to update
            enabled: Whether the rule should be enabled
            
        Returns:
            True if updated, False if not found
        """
        rule = self.rules_config.get_rule(rule_id)
        if rule:
            rule.enabled = enabled
            return True
        return False
    
    def update_rule(self, rule: AlertRule) -> bool:
        """
        Update an alert rule.
        
        Args:
            rule: Updated rule
            
        Returns:
            True if updated
        """
        self.rules_config.add_rule(rule)
        return True
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """
        Get all alert rules as dictionaries.
        
        Returns:
            List of rule dictionaries
        """
        return [rule.to_dict() for rule in self.rules_config.rules] 