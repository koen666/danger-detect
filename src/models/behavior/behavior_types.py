from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


class BehaviorType(Enum):
    """Enumeration of behavior types."""
    STATIC = 0          # 静止
    WALKING = 1         # 行走
    RUNNING = 2         # 奔跑
    FALLING = 3         # 跌倒
    FIGHTING = 4        # 打架
    GATHERING = 5       # 聚集
    WANDERING = 6       # 徘徊
    LOITERING = 7       # 徘徊/停留
    FOLLOWING = 8       # 跟随
    UNKNOWN = 9         # 未知行为


@dataclass
class BehaviorAnalysisResult:
    """Data class for behavior analysis results."""
    behavior_type: BehaviorType    # 行为类型
    confidence: float              # 置信度
    object_id: Optional[int] = None  # 关联的目标ID (如果可用)
    frame_idx: int = 0            # 帧索引
    data: Dict[str, Any] = None   # 附加数据
    timestamp: str = ""           # 时间戳
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "behavior_type": self.behavior_type.name,
            "confidence": self.confidence,
            "object_id": self.object_id,
            "frame_idx": self.frame_idx,
            "data": self.data,
            "timestamp": self.timestamp
        }


@dataclass
class Behavior:
    """Data class for behavior of an individual object."""
    object_id: int                     # 目标ID
    behavior_type: BehaviorType        # 行为类型
    confidence: float                  # 置信度
    position: Optional[Tuple[float, float]] = None  # 位置坐标 (x, y)
    timestamp: Optional[float] = None  # 时间戳
    metadata: Dict[str, Any] = None    # 元数据
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_id": self.object_id,
            "behavior_type": self.behavior_type.name,
            "confidence": self.confidence,
            "position": self.position,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class Interaction:
    """Data class for interaction between multiple objects."""
    object_ids: List[int]              # 交互的目标ID列表
    behavior_type: BehaviorType        # 交互类型
    confidence: float                  # 置信度
    position: Optional[Tuple[float, float]] = None  # 交互中心点
    timestamp: Optional[float] = None  # 时间戳
    metadata: Dict[str, Any] = None    # 元数据
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_ids": self.object_ids,
            "behavior_type": self.behavior_type.name,
            "confidence": self.confidence,
            "position": self.position,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        } 