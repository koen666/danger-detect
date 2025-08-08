import logging
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable

# Alert level enum for different severity levels
class AlertLevel(Enum):
    INFO = 0     # Informational events
    WARNING = 1  # Warnings that might need attention
    ALERT = 2    # Alerts that need immediate attention
    CRITICAL = 3 # Critical events requiring urgent action
    
    @classmethod
    def from_string(cls, level_str: str) -> 'AlertLevel':
        """Convert string to AlertLevel."""
        level_map = {
            'info': cls.INFO,
            'warning': cls.WARNING,
            'alert': cls.ALERT,
            'critical': cls.CRITICAL
        }
        return level_map.get(level_str.lower(), cls.INFO)


@dataclass
class AlertRule:
    """
    Alert rule definition for different detection scenarios.
    """
    id: str                    # Unique rule identifier
    name: str                  # Human-readable name
    description: str           # Rule description
    level: AlertLevel          # Alert severity level
    source_type: str           # Source of the alert (behavior, object, motion, etc.)
    conditions: Dict[str, Any] # Conditions that trigger the alert
    enabled: bool = True       # Whether the rule is enabled
    cooldown: int = 0          # Cooldown period in seconds to avoid alert flooding
    
    # Track last time this rule was triggered
    last_triggered: float = 0  
    
    # Count of times this rule has been triggered
    trigger_count: int = 0     
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'level': self.level.name,
            'source_type': self.source_type,
            'conditions': self.conditions,
            'enabled': self.enabled,
            'cooldown': self.cooldown,
            'trigger_count': self.trigger_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRule':
        """Create rule from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            description=data['description'],
            level=AlertLevel.from_string(data['level']),
            source_type=data['source_type'],
            conditions=data['conditions'],
            enabled=data.get('enabled', True),
            cooldown=data.get('cooldown', 0),
            trigger_count=data.get('trigger_count', 0)
        )


@dataclass
class AlertRuleConfig:
    """
    Configuration for alert rules.
    """
    rules: List[AlertRule] = field(default_factory=list)
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add a rule to the configuration."""
        # Check if rule with same ID already exists
        for i, existing_rule in enumerate(self.rules):
            if existing_rule.id == rule.id:
                # Replace existing rule
                self.rules[i] = rule
                return
        # Add new rule
        self.rules.append(rule)
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the configuration."""
        for i, rule in enumerate(self.rules):
            if rule.id == rule_id:
                self.rules.pop(i)
                return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get a rule by ID."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None
    
    def get_rules_by_source(self, source_type: str) -> List[AlertRule]:
        """Get all rules for a specific source type."""
        return [rule for rule in self.rules if rule.source_type == source_type and rule.enabled]
    
    def save_to_file(self, file_path: str) -> bool:
        """Save rules to a JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump([rule.to_dict() for rule in self.rules], f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Failed to save rules to {file_path}: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'AlertRuleConfig':
        """Load rules from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                rules_data = json.load(f)
            
            rules = [AlertRule.from_dict(rule_data) for rule_data in rules_data]
            return cls(rules=rules)
        except Exception as e:
            logging.error(f"Failed to load rules from {file_path}: {e}")
            return cls()
    
    def create_default_rules(self) -> None:
        """Create default rules for common scenarios."""
        default_rules = [
            AlertRule(
                id="intrusion_detection",
                name="Intrusion Detection",
                description="Detect when a person enters a restricted area",
                level=AlertLevel.ALERT,
                source_type="behavior",
                conditions={
                    "behavior_type": "INTRUSION",
                    "confidence_threshold": 0.7
                }
            ),
            AlertRule(
                id="loitering_detection",
                name="Loitering Detection",
                description="Detect when a person stays in an area for too long",
                level=AlertLevel.WARNING,
                source_type="behavior",
                conditions={
                    "behavior_type": "WANDERING",
                    "min_duration": 30,  # seconds
                    "confidence_threshold": 0.6
                }
            ),
            AlertRule(
                id="fast_motion_detection",
                name="Fast Motion Detection",
                description="Detect rapid movement in the scene",
                level=AlertLevel.INFO,
                source_type="motion",
                conditions={
                    "motion_threshold": 50,
                    "min_area": 1000
                }
            ),
            AlertRule(
                id="object_counting",
                name="Object Counting Threshold",
                description="Alert when too many objects are detected",
                level=AlertLevel.WARNING,
                source_type="object",
                conditions={
                    "class_name": "person",
                    "min_count": 5,
                    "confidence_threshold": 0.5
                }
            ),
            AlertRule(
                id="abandoned_object",
                name="Abandoned Object Detection",
                description="Detect stationary objects left behind",
                level=AlertLevel.WARNING,
                source_type="object",
                conditions={
                    "class_name": "suitcase|backpack|handbag",
                    "stationary_time": 60,  # seconds
                    "confidence_threshold": 0.6
                }
            ),
            AlertRule(
                id="abnormal_scene",
                name="Abnormal Scene Detection",
                description="Detect abnormal scenes using Qwen-VL",
                level=AlertLevel.ALERT,
                source_type="scene",
                conditions={
                    "normal_context": "people walking normally in a hallway",
                    "anomaly_threshold": 0.7
                }
            ),
            AlertRule(
                id="fighting_detection",
                name="Fighting Detection",
                description="Detect when people are fighting",
                level=AlertLevel.CRITICAL,
                source_type="behavior",
                conditions={
                    "behavior_type": "FIGHTING",
                    "confidence_threshold": 0.7
                }
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule) 