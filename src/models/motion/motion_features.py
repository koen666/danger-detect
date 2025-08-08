"""
运动特征模块 - 导入相关类以保持向后兼容性
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

from models.motion.motion_feature_base import MotionFeature, MotionFeatureExtractor
from models.motion.optical_flow import OpticalFlowExtractor
from models.motion.motion_history import MotionHistoryExtractor
from models.motion.motion_manager import MotionFeatureManager

# 为了保持向后兼容性，这个文件只是从其他模块导入相关类
# 实际实现已经分散到以下文件中:
# - models/motion/motion_feature_base.py (MotionFeature, MotionFeatureExtractor)
# - models/motion/optical_flow.py (OpticalFlowExtractor)
# - models/motion/motion_history.py (MotionHistoryExtractor)
# - models/motion/motion_manager.py (MotionFeatureManager) 