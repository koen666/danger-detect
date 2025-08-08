import os
import argparse
import cv2
import json
import time
import numpy as np
from typing import Dict, List, Any

from models.video_capture import VideoCaptureManager
from models.alert_system import AlertSystem
from models.alert_rule import AlertRule, AlertLevel, AlertRuleConfig
from models.alert_event import AlertEvent
from models.alert_processor import AlertConditionEvaluator
from models.alert_plugins import FileNotifier, NotificationManager


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test Alert System")
    
    # Input sources
    parser.add_argument("--source", type=str, default="0", help="Video source (camera index, file path, or URL)")
    
    # Alert system parameters
    parser.add_argument("--output_dir", type=str, default="alert_output", 
                        help="Output directory for alert data")
    parser.add_argument("--rules_file", type=str, default=None, 
                        help="Path to custom rules file (optional)")
    parser.add_argument("--simulate_alerts", action="store_true",
                        help="Simulate alerts at regular intervals for testing")
    parser.add_argument("--alert_interval", type=int, default=5,
                        help="Interval between simulated alerts (seconds)")
    
    # Display options
    parser.add_argument("--display", action="store_true", 
                        help="Display video with alerts")
    
    return parser.parse_args()


class AlertSystemTester:
    """Test class for the alert system."""
    
    def __init__(self, args):
        """
        Initialize the alert system tester.
        
        Args:
            args: Command-line arguments
        """
        self.args = args
        
        # Create output directory
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize video capture
        self.video_capture = VideoCaptureManager(
            source=args.source,
            target_width=640,
            target_height=480
        )
        
        # Initialize alert system
        self.init_alert_system()
        
        # Initialize state
        self.is_running = False
        self.frame_count = 0
        self.last_alert_time = 0
        self.simulated_alert_types = [
            ("motion", AlertLevel.INFO, "Motion detected in scene"),
            ("object", AlertLevel.WARNING, "Multiple persons detected"),
            ("behavior", AlertLevel.ALERT, "Loitering behavior detected"),
            ("scene", AlertLevel.CRITICAL, "Abnormal scene detected")
        ]
    
    def init_alert_system(self):
        """Initialize the alert system."""
        # Create configuration
        alert_config = {
            "output_dir": self.output_dir,
            "processing": {
                "process_every_n_frames": 5
            },
            "notifications": {
                "file": {
                    "type": "file",
                    "enabled": True,
                    "min_level": "info",
                    "output_file": os.path.join(self.output_dir, "alerts.log"),
                    "append": True
                }
            }
        }
        
        # Save config to file
        config_path = os.path.join(self.output_dir, "alert_config.json")
        with open(config_path, 'w') as f:
            json.dump(alert_config, f, indent=2)
        
        # Initialize alert system
        if self.args.rules_file:
            # Use custom rules file
            self.alert_system = AlertSystem(config_path)
            
            # Load custom rules
            custom_rules = AlertRuleConfig.load_from_file(self.args.rules_file)
            for rule in custom_rules.rules:
                self.alert_system.update_rule(rule)
        else:
            # Use default rules
            self.alert_system = AlertSystem(config_path)
        
        # Start the alert system
        self.alert_system.start()
    
    def run(self):
        """Run the alert system test."""
        if not self.video_capture.is_opened():
            print("Failed to open video source")
            return
        
        self.is_running = True
        start_time = time.time()
        
        print(f"Starting alert system test with source: {self.args.source}")
        print(f"Output directory: {self.output_dir}")
        print(f"Alert statistics on start: {self.alert_system.get_alert_stats()}")
        
        # Main loop
        try:
            while self.is_running and self.video_capture.is_opened():
                # Read frame
                ret, frame = self.video_capture.read()
                if not ret or frame is None:
                    print("End of video or error reading frame")
                    break
                
                # Process frame (generates alerts if simulating)
                alerts = self.process_frame(frame)
                
                # Display frame if requested
                if self.args.display:
                    self.display_frame(frame, alerts)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):  # ESC or q to quit
                        break
                    elif key == ord('a'):  # 'a' to generate a test alert
                        self.generate_test_alert(frame)
                
                # Update frame count
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            self.cleanup()
    
    def process_frame(self, frame):
        """
        Process a frame and generate alerts if simulating.
        
        Args:
            frame: Current frame
            
        Returns:
            List of alerts generated
        """
        alerts = []
        
        # Simulate alerts if enabled
        if self.args.simulate_alerts:
            current_time = time.time()
            if current_time - self.last_alert_time > self.args.alert_interval:
                # Generate a test alert
                alerts = [self.generate_test_alert(frame)]
                self.last_alert_time = current_time
        
        return alerts
    
    def generate_test_alert(self, frame):
        """
        Generate a test alert for demonstration purposes.
        
        Args:
            frame: Current frame
            
        Returns:
            Generated alert
        """
        # Select a random alert type
        source_type, level, message_prefix = self.simulated_alert_types[
            self.frame_count % len(self.simulated_alert_types)
        ]
        
        # Create alert details
        details = {
            "simulated": True,
            "frame_index": self.frame_count,
            "timestamp": time.time()
        }
        
        # Add type-specific details
        if source_type == "motion":
            details["motion_count"] = np.random.randint(10, 100)
            details["estimated_area"] = details["motion_count"] * 16 * 16
            message = f"{message_prefix} (area: {details['estimated_area']} px²)"
        
        elif source_type == "object":
            details["count"] = np.random.randint(3, 10)
            details["detected_objects"] = [i for i in range(details["count"])]
            message = f"{message_prefix} (count: {details['count']})"
        
        elif source_type == "behavior":
            details["behavior_type"] = "WANDERING"
            details["object_id"] = np.random.randint(1, 10)
            details["duration"] = np.random.randint(20, 120)
            message = f"{message_prefix} (ID: {details['object_id']}, {details['duration']}s)"
        
        else:  # scene
            details["anomaly_score"] = np.random.uniform(0.7, 1.0)
            details["caption"] = "Potential security threat detected in the scene"
            message = f"{message_prefix} ({details['anomaly_score']:.2f})"
        
        # Create alert event
        alert = AlertEvent.create(
            rule_id=f"test_{source_type}",
            level=level,
            source_type=source_type,
            message=message,
            details=details,
            frame_idx=self.frame_count,
            frame=frame
        )
        
        # Process the alert
        self.alert_system.event_store.add_event(alert)
        self.alert_system.notification_manager.notify(alert, synchronous=True)
        
        print(f"Generated alert: {alert.level.name} - {alert.message}")
        
        return alert
    
    def display_frame(self, frame, alerts):
        """
        Display frame with overlays.
        
        Args:
            frame: Current frame
            alerts: Alerts for this frame
        """
        # Create copy for display
        display_frame = frame.copy()
        
        # Get recent alerts
        recent_alerts = self.alert_system.get_recent_alerts(count=5)
        
        # Draw alert info
        h, w = display_frame.shape[:2]
        
        # Create overlay for alert box
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Draw title
        cv2.putText(display_frame, "ALERT SYSTEM TEST", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(display_frame, "Press 'q' to quit, 'a' to generate alert", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw stats
        stats = self.alert_system.get_alert_stats()
        stats_text = f"Alerts: {stats['total']} | INFO: {stats.get('by_level', {}).get('INFO', 0)} | "
        stats_text += f"WARNING: {stats.get('by_level', {}).get('WARNING', 0)} | "
        stats_text += f"ALERT: {stats.get('by_level', {}).get('ALERT', 0)} | "
        stats_text += f"CRITICAL: {stats.get('by_level', {}).get('CRITICAL', 0)}"
        
        cv2.putText(display_frame, stats_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw recent alerts
        y_pos = 150
        for i, alert_dict in enumerate(recent_alerts):
            level_name = alert_dict.get("level", "INFO")
            level_color = {
                "INFO": (255, 255, 0),     # Cyan
                "WARNING": (0, 165, 255),  # Orange
                "ALERT": (0, 0, 255),      # Red
                "CRITICAL": (0, 0, 255)    # Red
            }.get(level_name, (255, 255, 255))
            
            # Format time as HH:MM:SS
            timestamp_str = alert_dict.get("datetime", "")
            
            # Format message
            message = f"[{level_name}] {timestamp_str} - {alert_dict.get('message', '')}"
            cv2.putText(display_frame, message, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, level_color, 1)
            y_pos += 25
        
        # Show frame
        cv2.imshow("Alert System Test", display_frame)
    
    def cleanup(self):
        """Clean up resources."""
        print("Shutting down alert system test")
        
        # Stop alert system
        self.alert_system.stop()
        
        # Release video capture
        self.video_capture.release()
        
        # Close display windows
        if self.args.display:
            cv2.destroyAllWindows()
        
        # Print final stats
        print(f"Processed {self.frame_count} frames")
        print(f"Alert statistics on exit: {self.alert_system.get_alert_stats()}")
        print(f"Alert output directory: {self.output_dir}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Run the alert system test
    tester = AlertSystemTester(args)
    tester.run()


if __name__ == "__main__":
    main() 