#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频监控系统 - 工具函数包
包含各种辅助工具和实用函数
"""

from .frame_validation import validate_frame, is_frame_valid
from .motion_utils import calculate_motion_metrics, smooth_motion_data
from .preprocessing import preprocess_frame, normalize_frame