"""
轨迹分析模块 - 导入相关类以保持向后兼容性
"""

from models.trajectory import ObjectTrajectory
from models.trajectory_manager import TrajectoryManager
from models.interaction_detector import InteractionDetector

# 为了保持向后兼容性，这个文件只是从其他模块导入相关类
# 实际实现已经分散到以下文件中:
# - models/trajectory.py (ObjectTrajectory)
# - models/trajectory_manager.py (TrajectoryManager)
# - models/interaction_detector.py (InteractionDetector) 