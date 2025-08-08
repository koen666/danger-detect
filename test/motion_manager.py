#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运动特征提取模块 - 负责从视频帧中提取各种运动特征
"""

import cv2
import numpy as np
import logging
import time
from enum import Enum

# 配置日志
logger = logging.getLogger("MotionFeatureManager")

class FeatureType(Enum):
    """特征类型枚举"""
    OPTICAL_FLOW = "optical_flow"
    MOTION_HISTORY = "motion_history"
    BACKGROUND_SUB = "background_sub"
    CONTOUR = "contour"
    KEYPOINT = "keypoint"

class MotionFeatureManager:
    """运动特征管理器 - 负责从视频帧中提取各种运动特征"""
    
    def __init__(self, use_optical_flow=True, use_motion_history=False, 
                 use_background_sub=False, use_contour=True, use_keypoint=False,
                 optical_flow_method='farneback', use_gpu=False):
        """初始化运动特征管理器
        
        Args:
            use_optical_flow (bool): 是否使用光流
            use_motion_history (bool): 是否使用运动历史
            use_background_sub (bool): 是否使用背景减除
            use_contour (bool): 是否使用轮廓检测
            use_keypoint (bool): 是否使用关键点检测
            optical_flow_method (str): 光流方法 ('farneback', 'sparse', 'dense_pyr_lk')
            use_gpu (bool): 是否使用GPU加速
        """
        self.use_optical_flow = use_optical_flow
        self.use_motion_history = use_motion_history
        self.use_background_sub = use_background_sub
        self.use_contour = use_contour
        self.use_keypoint = use_keypoint
        self.optical_flow_method = optical_flow_method
        self.use_gpu = use_gpu
        
        # 初始化状态变量
        self.prev_gray = None
        self.motion_history = None
        self.bg_subtractor = None
        self.feature_detector = None
        self.frame_count = 0
        
        # 初始化背景减除器
        if self.use_background_sub:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=25, detectShadows=False)
        
        # 初始化特征检测器
        if self.use_keypoint:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0 and self.use_gpu:
                self.feature_detector = cv2.cuda.FastFeatureDetector.create(
                    threshold=20, nonmaxSuppression=True)
            else:
                self.feature_detector = cv2.FastFeatureDetector_create(
                    threshold=20, nonmaxSuppression=True)
        
        # 初始化光流参数
        if self.use_optical_flow:
            if self.optical_flow_method == 'sparse':
                # 用于LK光流的参数
                self.lk_params = dict(
                    winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                # 角点检测参数
                self.feature_params = dict(
                    maxCorners=100,
                    qualityLevel=0.3,
                    minDistance=7,
                    blockSize=7)
                self.prev_points = None
            elif self.optical_flow_method == 'farneback':
                # Farneback光流参数
                self.farneback_params = dict(
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0)
            elif self.optical_flow_method == 'dense_pyr_lk':
                # 密集金字塔LK
                self.lk_pyr_params = dict(
                    winSize=(15, 15),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        logger.info(f"初始化运动特征管理器: 光流={use_optical_flow}({optical_flow_method}), " 
                    f"运动历史={use_motion_history}, 背景减除={use_background_sub}, "
                    f"轮廓={use_contour}, 关键点={use_keypoint}, GPU={use_gpu}")
    
    def extract_features(self, frame, prev_frame=None):
        """从帧中提取特征
        
        Args:
            frame: 当前帧
            prev_frame: 前一帧（可选）
        
        Returns:
            features: 提取的特征字典
        """
        self.frame_count += 1
        features = {}
        
        # 转换为灰度图
        if frame is None:
            return features
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features['gray'] = gray
        
        # 提取光流
        if self.use_optical_flow and self.prev_gray is not None:
            start_time = time.time()
            if self.optical_flow_method == 'farneback':
                flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, **self.farneback_params)
                features[FeatureType.OPTICAL_FLOW.value] = flow
                
                # 计算光流统计信息
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                features['flow_magnitude'] = mag
                features['flow_angle'] = ang
                features['flow_mean_magnitude'] = np.mean(mag)
                features['flow_max_magnitude'] = np.max(mag)
                
                # 计算运动矢量
                motion_vectors = []
                step = 16  # 采样步长
                for y in range(0, gray.shape[0], step):
                    for x in range(0, gray.shape[1], step):
                        fx, fy = flow[y, x]
                        if mag[y, x] > 1.0:  # 只保留显著运动
                            motion_vectors.append((x, y, fx, fy, mag[y, x]))
                features['motion_vectors'] = motion_vectors
                
            elif self.optical_flow_method == 'sparse':
                if self.prev_points is None:
                    self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)
                
                if self.prev_points is not None and len(self.prev_points) > 0:
                    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, self.prev_points, None, **self.lk_params)
                    
                    if next_points is not None:
                        # 选择好的点
                        good_old = self.prev_points[status == 1]
                        good_new = next_points[status == 1]
                        
                        features[FeatureType.OPTICAL_FLOW.value] = (good_old, good_new)
                        
                        # 计算运动矢量
                        motion_vectors = []
                        if len(good_old) > 0:
                            # 计算位移向量
                            displacements = good_new - good_old
                            magnitudes = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
                            
                            for i, ((x1, y1), (x2, y2), mag) in enumerate(zip(good_old, good_new, magnitudes)):
                                if mag > 1.0:  # 只保留显著运动
                                    motion_vectors.append((int(x1), int(y1), x2-x1, y2-y1, mag))
                            
                            features['flow_mean_magnitude'] = np.mean(magnitudes)
                            features['flow_max_magnitude'] = np.max(magnitudes)
                        
                        features['motion_vectors'] = motion_vectors
                        
                        # 更新点
                        self.prev_points = good_new.reshape(-1, 1, 2)
                
                # 定期刷新特征点
                if self.frame_count % 10 == 0 or self.prev_points is None or len(self.prev_points) < 10:
                    self.prev_points = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            
            features['flow_extraction_time'] = time.time() - start_time
        
        # 提取运动历史
        if self.use_motion_history:
            start_time = time.time()
            # 计算帧差
            if prev_frame is not None:
                prev_gray_for_diff = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(gray, prev_gray_for_diff)
                _, motion_mask = cv2.threshold(frame_diff, 25, 1, cv2.THRESH_BINARY)
                
                # 更新运动历史图
                if self.motion_history is None:
                    self.motion_history = np.zeros_like(gray, dtype=np.float32)
                
                timestamp = self.frame_count / 30.0  # 假设30fps
                cv2.motempl.updateMotionHistory(motion_mask, self.motion_history, timestamp, 0.5)  # 0.5s的历史
                
                # 计算运动梯度
                mg_mask, mg_orient = cv2.motempl.calcMotionGradient(
                    self.motion_history, 0.25, 0.05, apertureSize=3)
                
                # 分割运动区域
                seg_mask, _ = cv2.motempl.segmentMotion(
                    self.motion_history, timestamp, 0.25)
                
                features[FeatureType.MOTION_HISTORY.value] = self.motion_history
                features['motion_gradient_mask'] = mg_mask
                features['motion_gradient_orient'] = mg_orient
                features['motion_segments'] = seg_mask
            
            features['mhi_extraction_time'] = time.time() - start_time
        
        # 背景减除
        if self.use_background_sub:
            start_time = time.time()
            fg_mask = self.bg_subtractor.apply(frame)
            features[FeatureType.BACKGROUND_SUB.value] = fg_mask
            
            # 计算前景比例
            fg_ratio = np.sum(fg_mask > 0) / (fg_mask.shape[0] * fg_mask.shape[1])
            features['foreground_ratio'] = fg_ratio
            
            features['bg_extraction_time'] = time.time() - start_time
        
        # 轮廓检测
        if self.use_contour:
            start_time = time.time()
            
            # 使用帧差或背景减除的结果
            if 'motion_mask' in locals():
                mask = motion_mask
            elif self.use_background_sub and FeatureType.BACKGROUND_SUB.value in features:
                mask = features[FeatureType.BACKGROUND_SUB.value]
            else:
                # 直接从当前帧计算
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
                mask = thresh
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤小轮廓
            valid_contours = []
            for c in contours:
                if cv2.contourArea(c) > 50:  # 面积阈值
                    valid_contours.append(c)
            
            features[FeatureType.CONTOUR.value] = valid_contours
            features['contour_count'] = len(valid_contours)
            
            # 计算轮廓面积总和
            total_area = 0
            frame_area = gray.shape[0] * gray.shape[1]
            for c in valid_contours:
                area = cv2.contourArea(c)
                total_area += area
            
            features['contour_area_ratio'] = total_area / frame_area if frame_area > 0 else 0
            
            features['contour_extraction_time'] = time.time() - start_time
        
        # 关键点检测
        if self.use_keypoint:
            start_time = time.time()
            
            if self.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # GPU加速
                gray_gpu = cv2.cuda_GpuMat()
                gray_gpu.upload(gray)
                keypoints_gpu = self.feature_detector.detect(gray_gpu)
                keypoints = cv2.cuda_KeyPoint.convert(keypoints_gpu)
            else:
                # CPU
                keypoints = self.feature_detector.detect(gray)
            
            features[FeatureType.KEYPOINT.value] = keypoints
            features['keypoint_count'] = len(keypoints)
            
            # 计算关键点分布统计
            if len(keypoints) > 0:
                pts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
                mean_pt = np.mean(pts, axis=0)
                std_pt = np.std(pts, axis=0)
                features['keypoint_mean_pos'] = mean_pt
                features['keypoint_std_pos'] = std_pt
            
            features['keypoint_extraction_time'] = time.time() - start_time
        
        # 更新前一帧
        self.prev_gray = gray
        
        return features
    
    def visualize_features(self, frame, features):
        """可视化提取的特征
        
        Args:
            frame: 原始帧
            features: 提取的特征
        
        Returns:
            vis_frame: 可视化后的帧
        """
        vis_frame = frame.copy()
        
        # 可视化光流
        if self.use_optical_flow and FeatureType.OPTICAL_FLOW.value in features:
            flow = features[FeatureType.OPTICAL_FLOW.value]
            
            if self.optical_flow_method == 'farneback' and isinstance(flow, np.ndarray):
                # 绘制光流场
                if 'motion_vectors' in features:
                    for x, y, fx, fy, mag in features['motion_vectors']:
                        # 按照运动大小确定颜色（红色到绿色）
                        color_val = min(255, int(mag * 10))
                        color = (0, color_val, 255 - color_val)
                        cv2.arrowedLine(vis_frame, (x, y), (int(x + fx), int(y + fy)), color, 1, tipLength=0.3)
            
            elif self.optical_flow_method == 'sparse' and isinstance(flow, tuple):
                good_old, good_new = flow
                # 绘制轨迹
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    # 计算位移大小，用于确定颜色
                    mag = np.sqrt((a - c)**2 + (b - d)**2)
                    color_val = min(255, int(mag * 10))
                    color = (0, color_val, 255 - color_val)
                    cv2.line(vis_frame, (int(c), int(d)), (int(a), int(b)), color, 2)
                    cv2.circle(vis_frame, (int(a), int(b)), 3, color, -1)
        
        # 可视化轮廓
        if self.use_contour and FeatureType.CONTOUR.value in features:
            contours = features[FeatureType.CONTOUR.value]
            cv2.drawContours(vis_frame, contours, -1, (0, 255, 0), 1)
            
            # 绘制轮廓的边界框
            for c in contours:
                area = cv2.contourArea(c)
                if area > 200:  # 只显示较大的轮廓
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
        
        # 可视化关键点
        if self.use_keypoint and FeatureType.KEYPOINT.value in features:
            keypoints = features[FeatureType.KEYPOINT.value]
            vis_frame = cv2.drawKeypoints(vis_frame, keypoints, None, color=(0, 0, 255),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # 在顶部添加文本信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 20
        
        # 显示光流统计
        if 'flow_mean_magnitude' in features:
            cv2.putText(vis_frame, f"Flow Avg: {features['flow_mean_magnitude']:.2f}", 
                      (10, y_offset), font, 0.5, (0, 255, 255), 1)
            y_offset += 20
        
        # 显示轮廓信息
        if 'contour_count' in features:
            cv2.putText(vis_frame, f"Contours: {features['contour_count']}", 
                      (10, y_offset), font, 0.5, (0, 255, 0), 1)
            y_offset += 20
        
        if 'contour_area_ratio' in features:
            area_percent = features['contour_area_ratio'] * 100
            cv2.putText(vis_frame, f"Motion Area: {area_percent:.1f}%", 
                      (10, y_offset), font, 0.5, (0, 255, 0), 1)
            y_offset += 20
        
        # 显示关键点数量
        if 'keypoint_count' in features:
            cv2.putText(vis_frame, f"Keypoints: {features['keypoint_count']}", 
                      (10, y_offset), font, 0.5, (0, 0, 255), 1)
        
        return vis_frame
    
    def reset(self):
        """重置状态"""
        self.prev_gray = None
        self.motion_history = None
        if self.use_background_sub:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=25, detectShadows=False)
        self.prev_points = None
        self.frame_count = 0
        logger.info("重置运动特征管理器状态")
    
    def get_feature_types(self):
        """获取已启用的特征类型"""
        enabled_features = []
        if self.use_optical_flow:
            enabled_features.append(FeatureType.OPTICAL_FLOW)
        if self.use_motion_history:
            enabled_features.append(FeatureType.MOTION_HISTORY)
        if self.use_background_sub:
            enabled_features.append(FeatureType.BACKGROUND_SUB)
        if self.use_contour:
            enabled_features.append(FeatureType.CONTOUR)
        if self.use_keypoint:
            enabled_features.append(FeatureType.KEYPOINT)
        return enabled_features 