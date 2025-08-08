import abc
import smtplib
import requests
import logging
import os
import time
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Dict, List, Any, Optional, Callable

from models.alert_event import AlertEvent
from models.alert_rule import AlertLevel


class AlertNotifier(abc.ABC):
    """Abstract base class for alert notification plugins."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the notifier with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = "BaseNotifier"
        self.enabled = config.get("enabled", True)
        self.min_level = AlertLevel.from_string(config.get("min_level", "info"))
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"AlertNotifier.{self.name}")
    
    @abc.abstractmethod
    def notify(self, event: AlertEvent) -> bool:
        """
        Send notification for an alert event.
        
        Args:
            event: Alert event to notify about
            
        Returns:
            True if notification was sent successfully, False otherwise
        """
        pass
    
    def should_notify(self, event: AlertEvent) -> bool:
        """
        Check if this notifier should handle this event.
        
        Args:
            event: Alert event to check
            
        Returns:
            True if the notifier should handle this event, False otherwise
        """
        # Only notify if enabled and event level is >= min_level
        return self.enabled and event.level.value >= self.min_level.value


class EmailNotifier(AlertNotifier):
    """Email notification plugin."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the email notifier.
        
        Args:
            config: Configuration dictionary with the following keys:
                smtp_server: SMTP server address
                smtp_port: SMTP server port
                smtp_username: SMTP username
                smtp_password: SMTP password
                from_addr: Sender email address
                to_addrs: List of recipient email addresses
                use_ssl: Whether to use SSL
                enabled: Whether the notifier is enabled
                min_level: Minimum alert level to notify about
        """
        super().__init__(config)
        self.name = "EmailNotifier"
        
        # Email-specific config
        self.smtp_server = config.get("smtp_server", "")
        self.smtp_port = config.get("smtp_port", 587)
        self.smtp_username = config.get("smtp_username", "")
        self.smtp_password = config.get("smtp_password", "")
        self.from_addr = config.get("from_addr", "")
        self.to_addrs = config.get("to_addrs", [])
        self.use_ssl = config.get("use_ssl", False)
        
        # Validate required fields
        if not all([self.smtp_server, self.smtp_username, 
                    self.smtp_password, self.from_addr, self.to_addrs]):
            self.logger.error("Email notifier configuration is incomplete")
            self.enabled = False
    
    def notify(self, event: AlertEvent) -> bool:
        """Send an email notification."""
        if not self.should_notify(event):
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = ", ".join(self.to_addrs)
            msg['Subject'] = f"{event.level.name} Alert: {event.message}"
            
            # Create HTML body
            html = f"""
            <html>
            <head></head>
            <body>
                <h2>{event.level.name} Alert: {event.message}</h2>
                <p><strong>Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}</p>
                <p><strong>Source:</strong> {event.source_type}</p>
                <p><strong>Rule:</strong> {event.rule_id}</p>
                <h3>Details:</h3>
                <pre>{str(event.details)}</pre>
            </body>
            </html>
            """
            
            # Attach HTML body
            msg.attach(MIMEText(html, 'html'))
            
            # Attach thumbnail if available
            if event.thumbnail is not None:
                import cv2
                _, img_data = cv2.imencode('.jpg', event.thumbnail)
                image = MIMEImage(img_data.tobytes())
                image.add_header('Content-ID', '<image1>')
                msg.attach(image)
            
            # Connect to server and send
            server = None
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
            
            server.login(self.smtp_username, self.smtp_password)
            server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            server.quit()
            
            self.logger.info(f"Sent email notification for event {event.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False


class WebhookNotifier(AlertNotifier):
    """Webhook notification plugin."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the webhook notifier.
        
        Args:
            config: Configuration dictionary with the following keys:
                webhook_url: Webhook URL
                headers: HTTP headers to send
                include_thumbnail: Whether to include thumbnail image
                enabled: Whether the notifier is enabled
                min_level: Minimum alert level to notify about
        """
        super().__init__(config)
        self.name = "WebhookNotifier"
        
        # Webhook-specific config
        self.webhook_url = config.get("webhook_url", "")
        self.headers = config.get("headers", {})
        self.include_thumbnail = config.get("include_thumbnail", True)
        
        # Validate required fields
        if not self.webhook_url:
            self.logger.error("Webhook URL is required")
            self.enabled = False
    
    def notify(self, event: AlertEvent) -> bool:
        """Send a webhook notification."""
        if not self.should_notify(event):
            return False
        
        try:
            # Prepare payload
            payload = event.to_dict(include_images=self.include_thumbnail)
            
            # Send webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            
            # Check response
            if response.status_code < 400:
                self.logger.info(f"Sent webhook notification for event {event.id}")
                return True
            else:
                self.logger.error(f"Webhook returned error {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            return False


class FileNotifier(AlertNotifier):
    """File-based notification plugin that writes alerts to a log file."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the file notifier.
        
        Args:
            config: Configuration dictionary with the following keys:
                output_file: Path to output file
                append: Whether to append to file or overwrite
                enabled: Whether the notifier is enabled
                min_level: Minimum alert level to notify about
        """
        super().__init__(config)
        self.name = "FileNotifier"
        
        # File-specific config
        self.output_file = config.get("output_file", "alerts.log")
        self.append = config.get("append", True)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
        
        # Write header if file doesn't exist or not appending
        if not os.path.exists(self.output_file) or not self.append:
            with open(self.output_file, 'w') as f:
                f.write("# Alert Log\n")
                f.write("# Timestamp | Level | Source | Message\n")
                f.write("# ----------------------------------------\n")
    
    def notify(self, event: AlertEvent) -> bool:
        """Write alert to log file."""
        if not self.should_notify(event):
            return False
        
        try:
            # Format log entry
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))
            log_entry = f"{timestamp} | {event.level.name} | {event.source_type} | {event.message}\n"
            
            # Write to file
            with open(self.output_file, 'a') as f:
                f.write(log_entry)
            
            self.logger.info(f"Wrote alert to file for event {event.id}")
            return True
                
        except Exception as e:
            self.logger.error(f"Failed to write alert to file: {e}")
            return False


class NotificationManager:
    """
    Manages multiple notification plugins and handles alert distribution.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the notification manager.
        
        Args:
            config: Configuration dictionary
        """
        self.notifiers = []
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("NotificationManager")
        
        # Create default config if none provided
        if config is None:
            config = {
                "file": {
                    "type": "file",
                    "enabled": True,
                    "min_level": "info",
                    "output_file": "alerts.log",
                    "append": True
                }
            }
        
        # Initialize notifiers from config
        self._init_notifiers(config)
        
        # Create notification thread
        self.notification_queue = []
        self.queue_lock = threading.Lock()
        self.notification_thread = threading.Thread(target=self._notification_worker, daemon=True)
        self.running = True
        self.notification_thread.start()
    
    def _init_notifiers(self, config: Dict[str, Any]):
        """Initialize notification plugins from config."""
        for name, notifier_config in config.items():
            notifier_type = notifier_config.get("type", "").lower()
            
            try:
                if notifier_type == "email":
                    self.notifiers.append(EmailNotifier(notifier_config))
                elif notifier_type == "webhook":
                    self.notifiers.append(WebhookNotifier(notifier_config))
                elif notifier_type == "file":
                    self.notifiers.append(FileNotifier(notifier_config))
                else:
                    self.logger.warning(f"Unknown notifier type: {notifier_type}")
            except Exception as e:
                self.logger.error(f"Failed to initialize notifier {name}: {e}")
        
        self.logger.info(f"Initialized {len(self.notifiers)} notification plugins")
    
    def add_notifier(self, notifier: AlertNotifier):
        """Add a notifier to the manager."""
        self.notifiers.append(notifier)
    
    def notify(self, event: AlertEvent, synchronous: bool = False):
        """
        Send notifications for an alert event.
        
        Args:
            event: Alert event to notify about
            synchronous: Whether to send notifications synchronously
        """
        if synchronous:
            # Send notifications immediately
            for notifier in self.notifiers:
                if notifier.should_notify(event):
                    notifier.notify(event)
        else:
            # Queue for background processing
            with self.queue_lock:
                self.notification_queue.append(event)
    
    def _notification_worker(self):
        """Background worker that processes the notification queue."""
        while self.running:
            # Process queue
            events_to_process = []
            with self.queue_lock:
                if self.notification_queue:
                    events_to_process = self.notification_queue
                    self.notification_queue = []
            
            # Process events
            for event in events_to_process:
                for notifier in self.notifiers:
                    try:
                        if notifier.should_notify(event):
                            notifier.notify(event)
                    except Exception as e:
                        self.logger.error(f"Error in notifier {notifier.name}: {e}")
            
            # Sleep if queue is empty
            if not events_to_process:
                time.sleep(0.5)
    
    def shutdown(self):
        """Shut down the notification manager."""
        self.running = False
        if self.notification_thread.is_alive():
            self.notification_thread.join(timeout=2.0)
        
        # Process any remaining notifications
        with self.queue_lock:
            events_to_process = self.notification_queue
            self.notification_queue = []
        
        # Process remaining events synchronously
        for event in events_to_process:
            for notifier in self.notifiers:
                try:
                    if notifier.should_notify(event):
                        notifier.notify(event)
                except Exception:
                    pass 