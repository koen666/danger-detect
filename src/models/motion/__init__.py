#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频监控系统 - 运动特征提取模块
用于从视频帧中提取运动特征和模式
"""

from .motion_manager import MotionFeatureManager
from .optical_flow import OpticalFlowExtractor
from .motion_history import MotionHistoryExtractor
from .motion_feature_base import MotionFeature, MotionFeatureExtractor

__all__ = [
    'MotionFeatureManager',
    'OpticalFlowExtractor',
    'MotionHistoryExtractor',
    'MotionFeature',
    'MotionFeatureExtractor'
] 