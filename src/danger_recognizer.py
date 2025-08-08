#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
危险行为识别模块 - 提供用于生产环境的危险行为检测功能
"""

import logging
import numpy as np
import cv2
import time
import os
from datetime import datetime
from threading import Lock

logger = logging.getLogger("DangerRecognizer")

class DangerRecognizer:
    """危险行为识别器 - 用于检测和识别视频中的危险行为"""
    
    DANGER_TYPES = {
        'sudden_motion': '突然剧烈运动',
        'large_area_motion': '大范围异常运动',
        'fall': '可能摔倒',
        'abnormal_pattern': '异常移动模式',
        'intrusion': '入侵警告区域',
        'loitering': '可疑徘徊',
    }
    
    def __init__(self, config=None):
        """初始化危险行为识别器
        
        Args:
            config: 配置字典，包含检测参数和阈值
        """
        # 默认配置
        self.config = {
            'feature_count_threshold': 100,  # 特征点数量阈值
            'feature_change_ratio': 1.5,     # 特征变化比率阈值
            'motion_magnitude_threshold': 10, # 运动幅度阈值
            'motion_area_threshold': 0.05,   # 运动区域阈值(占比)
            'fall_motion_threshold': 10,     # 摔倒检测阈值
            'alert_cooldown': 10,            # 告警冷却帧数
            'history_length': 30,            # 历史帧数量
            'save_alerts': True,             # 是否保存告警
            'alert_dir': 'alerts',           # 告警保存目录
            'min_confidence': 0.5,           # 最小置信度
        }
        
        # 更新用户配置
        if config:
            self.config.update(config)
        
        # 创建告警保存目录
        if self.config['save_alerts']:
            os.makedirs(self.config['alert_dir'], exist_ok=True)
        
        # 初始化状态
        self.history = []
        self.current_frame = 0
        self.last_alert_frame = 0
        self.alerts_count = {danger_type: 0 for danger_type in self.DANGER_TYPES.values()}
        self.last_features_count = 0
        self.debug_info = {}
        self.alert_lock = Lock()
        
        # 添加用于ROI区域的属性
        self.alert_regions = []  # 告警区域列表
        
        logger.info(f"危险行为识别器已初始化，特征点阈值:{self.config['feature_count_threshold']}, " + 
                   f"变化率阈值:{self.config['feature_change_ratio']}")
    
    def add_alert_region(self, region, name="警戒区"):
        """添加告警区域
        
        Args:
            region: [(x1,y1), (x2,y2), ...] 形式的多边形区域
            name: 区域名称
        """
        self.alert_regions.append({
            'points': np.array(region, dtype=np.int32),
            'name': name,
            'color': (0, 0, 255),
            'thickness': 2
        })
        logger.info(f"已添加告警区域: {name}")
        return len(self.alert_regions) - 1  # 返回区域ID
    
    def process_frame(self, frame, features, object_detections=None):
        """处理视频帧，分析是否存在危险行为
        
        Args:
            frame: 当前视频帧
            features: 从运动特征管理器获取的特征列表
            object_detections: 可选的物体检测结果列表
            
        Returns:
            alerts: 检测到的告警列表
        """
        self.current_frame += 1
        frame_shape = frame.shape if frame is not None else (480, 640, 3)
        
        # 提取当前帧的运动统计
        motion_stats = self._extract_motion_stats(features, (frame_shape[1], frame_shape[0]))
        
        # 更新特征计数
        feature_count = len(features) if features else 0
        self.debug_info = {
            'frame': self.current_frame,
            'feature_count': feature_count,
            'avg_magnitude': motion_stats['avg_magnitude'],
            'max_magnitude': motion_stats['max_magnitude'],
            'motion_area': motion_stats['motion_area'],
        }
        
        # 更新历史记录
        self.history.append(motion_stats)
        if len(self.history) > self.config['history_length']:
            self.history.pop(0)
        
        # 分析危险行为
        alerts = self._analyze_danger(frame, features, object_detections)
        
        # 更新统计信息
        with self.alert_lock:
            for alert in alerts:
                if alert['type'] in self.alerts_count:
                    self.alerts_count[alert['type']] += 1
                
                # 保存告警帧
                if self.config['save_alerts'] and frame is not None:
                    self._save_alert_frame(frame, alert)
        
        # 更新上一帧的特征计数
        self.last_features_count = feature_count
        
        return alerts
    
    def _extract_motion_stats(self, features, frame_size):
        """从特征中提取运动统计
        
        Args:
            features: 特征列表
            frame_size: 帧大小 (width, height)
            
        Returns:
            stats: 统计信息字典
        """
        frame_area = frame_size[0] * frame_size[1]
        stats = {
            'timestamp': time.time(),
            'avg_magnitude': 0,
            'max_magnitude': 0,
            'motion_directions': {},
            'motion_area': 0,
            'vertical_motion': 0,
            'feature_count': len(features) if features else 0,
        }
        
        if not features:
            return stats
        
        # 计算运动统计
        magnitudes = []
        motion_area = 0
        vertical_motion = 0
        
        for feature in features:
            if hasattr(feature, 'magnitude'):
                magnitudes.append(feature.magnitude)
            
            if hasattr(feature, 'position') and hasattr(feature, 'end_position'):
                # 计算方向
                dx = feature.end_position[0] - feature.position[0]
                dy = feature.end_position[1] - feature.position[1]
                
                vertical_motion += dy
                
                # 量化方向
                angle = np.degrees(np.arctan2(dy, dx))
                direction = round(angle / 45) * 45
                if direction in stats['motion_directions']:
                    stats['motion_directions'][direction] += 1
                else:
                    stats['motion_directions'][direction] = 1
                
                # 检查是否在告警区域内
                if self.alert_regions and hasattr(feature, 'position'):
                    for region in self.alert_regions:
                        if cv2.pointPolygonTest(region['points'], 
                                               (int(feature.position[0]), int(feature.position[1])), 
                                               False) >= 0:
                            stats['in_alert_region'] = True
                            break
            
            # 估算运动区域
            if hasattr(feature, 'magnitude'):
                motion_area += feature.magnitude * 10
        
        if magnitudes:
            stats['avg_magnitude'] = sum(magnitudes) / len(magnitudes)
            stats['max_magnitude'] = max(magnitudes)
        
        stats['motion_area'] = min(1.0, motion_area / frame_area)
        stats['vertical_motion'] = vertical_motion / max(1, len(features))
        
        return stats
    
    def _analyze_danger(self, frame, features, object_detections=None):
        """分析危险行为
        
        Args:
            frame: 当前视频帧
            features: 特征列表
            object_detections: 物体检测结果
            
        Returns:
            alerts: 告警列表
        """
        if len(self.history) < 3:
            return []
        
        alerts = []
        feature_count = len(features) if features else 0
        
        # 检查冷却时间
        if self.current_frame - self.last_alert_frame <= self.config['alert_cooldown']:
            return []
        
        # 1. 特征数量检测
        if feature_count > self.config['feature_count_threshold']:
            confidence = min(1.0, feature_count / (self.config['feature_count_threshold'] * 3))
            if confidence >= self.config['min_confidence']:
                alerts.append({
                    'type': self.DANGER_TYPES['sudden_motion'],
                    'confidence': confidence,
                    'frame': self.current_frame,
                    'feature_count': feature_count,
                    'threshold': self.config['feature_count_threshold'],
                })
        
        # 2. 特征变化率检测
        if self.last_features_count > 0:
            feature_change_ratio = feature_count / max(1, self.last_features_count)
            if (feature_change_ratio > self.config['feature_change_ratio'] and 
                feature_count > self.config['feature_count_threshold'] / 2):
                
                confidence = min(1.0, (feature_change_ratio - 1) / self.config['feature_change_ratio'])
                if confidence >= self.config['min_confidence']:
                    alerts.append({
                        'type': self.DANGER_TYPES['sudden_motion'],
                        'confidence': confidence,
                        'frame': self.current_frame,
                        'change_ratio': feature_change_ratio,
                        'threshold': self.config['feature_change_ratio'],
                    })
        
        # 3. 运动幅度检测
        current = self.history[-1]['avg_magnitude']
        if len(self.history) >= 5:
            prev_avg = sum(h['avg_magnitude'] for h in self.history[-6:-1]) / 5
            magnitude_ratio = current / max(0.1, prev_avg)
            
            if (magnitude_ratio > 1.2 and 
                current > self.config['motion_magnitude_threshold']):
                
                confidence = min(1.0, current / (self.config['motion_magnitude_threshold'] * 2))
                if confidence >= self.config['min_confidence']:
                    alerts.append({
                        'type': self.DANGER_TYPES['sudden_motion'],
                        'confidence': confidence,
                        'frame': self.current_frame,
                        'magnitude': current,
                        'threshold': self.config['motion_magnitude_threshold'],
                    })
        
        # 4. 大面积运动检测
        motion_area = self.history[-1]['motion_area']
        if motion_area > self.config['motion_area_threshold']:
            confidence = min(1.0, motion_area / self.config['motion_area_threshold'])
            if confidence >= self.config['min_confidence']:
                alerts.append({
                    'type': self.DANGER_TYPES['large_area_motion'],
                    'confidence': confidence,
                    'frame': self.current_frame,
                    'area': motion_area,
                    'threshold': self.config['motion_area_threshold'],
                })
        
        # 5. 检测警戒区域入侵
        if object_detections and self.alert_regions:
            for obj in object_detections:
                if 'bbox' in obj:  # 确保对象有边界框
                    x1, y1, x2, y2 = obj['bbox']
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    for region_idx, region in enumerate(self.alert_regions):
                        if cv2.pointPolygonTest(region['points'], (center_x, center_y), False) >= 0:
                            # 目标在警戒区域内
                            alerts.append({
                                'type': self.DANGER_TYPES['intrusion'],
                                'confidence': obj.get('confidence', 0.8),
                                'frame': self.current_frame,
                                'object': obj.get('class', 'unknown'),
                                'region': region_idx,
                                'region_name': region['name'],
                            })
        
        # 6. 摔倒检测 (简化版)
        if len(self.history) >= 10:
            vertical_motion = sum(h['vertical_motion'] for h in self.history[-10:-5])
            if vertical_motion > self.config['fall_motion_threshold']:
                recent_vertical = sum(h['vertical_motion'] for h in self.history[-3:])
                
                if abs(recent_vertical) < 5:  # 快速下降后静止
                    alerts.append({
                        'type': self.DANGER_TYPES['fall'],
                        'confidence': 0.7,
                        'frame': self.current_frame,
                        'motion': vertical_motion,
                        'threshold': self.config['fall_motion_threshold'],
                    })
        
        # 如果有告警，更新最后告警帧
        if alerts:
            self.last_alert_frame = self.current_frame
        
        return alerts
    
    def _save_alert_frame(self, frame, alert):
        """保存告警帧
        
        Args:
            frame: 当前视频帧
            alert: 告警信息
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{alert['type']}_{self.current_frame}_{timestamp}.jpg"
        filepath = os.path.join(self.config['alert_dir'], filename)
        
        # 在图像上添加告警信息
        vis_frame = frame.copy()
        
        # 绘制告警信息
        alert_text = f"{alert['type']} ({alert['confidence']:.2f})"
        cv2.putText(vis_frame, alert_text, (10, vis_frame.shape[0] - 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 添加红色边框
        cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1], vis_frame.shape[0]), (0, 0, 255), 3)
        
        # 添加更多的调试信息
        y_offset = 30
        for key, value in self.debug_info.items():
            text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # 绘制警戒区域
        for region in self.alert_regions:
            cv2.polylines(vis_frame, [region['points']], True, region['color'], region['thickness'])
        
        # 保存图像
        cv2.imwrite(filepath, vis_frame)
        logger.info(f"已保存告警帧: {filepath}")
    
    def visualize(self, frame, alerts=None, features=None, show_debug=True):
        """可视化危险行为检测结果
        
        Args:
            frame: 当前视频帧
            alerts: 告警列表
            features: 特征列表
            show_debug: 是否显示调试信息
            
        Returns:
            vis_frame: 可视化后的视频帧
        """
        if frame is None:
            return None
        
        vis_frame = frame.copy()
        
        # 绘制警戒区域
        for region in self.alert_regions:
            cv2.polylines(vis_frame, [region['points']], True, region['color'], region['thickness'])
            # 添加区域名称
            x, y = region['points'].mean(axis=0).astype(int)
            cv2.putText(vis_frame, region['name'], (x, y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, region['color'], 1)
        
        # 绘制告警信息
        if alerts:
            # 添加红色边框
            cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1], vis_frame.shape[0]), (0, 0, 255), 3)
            
            # 显示告警文本
            for i, alert in enumerate(alerts):
                alert_text = f"{alert['type']} ({alert['confidence']:.2f})"
                cv2.putText(vis_frame, alert_text, (10, vis_frame.shape[0] - 30 - i*30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示调试信息
        if show_debug:
            y_offset = 30
            for key, value in self.debug_info.items():
                text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
                cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            # 显示告警统计
            y_offset = 30
            for i, (alert_type, count) in enumerate(self.alerts_count.items()):
                if count > 0:
                    cv2.putText(vis_frame, f"{alert_type}: {count}", (vis_frame.shape[1] - 180, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    y_offset += 20
        
        return vis_frame
    
    def get_alert_stats(self):
        """获取告警统计
        
        Returns:
            stats: 告警统计信息
        """
        with self.alert_lock:
            return self.alerts_count.copy()
    
    def reset_stats(self):
        """重置告警统计"""
        with self.alert_lock:
            self.alerts_count = {danger_type: 0 for danger_type in self.DANGER_TYPES.values()}
    
    def reset(self):
        """重置危险行为识别器"""
        with self.alert_lock:
            self.history = []
            self.current_frame = 0
            self.last_alert_frame = 0
            self.alerts_count = {danger_type: 0 for danger_type in self.DANGER_TYPES.values()}
            self.last_features_count = 0
            logger.info("危险行为识别器已重置")

# 简单测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建危险行为识别器
    recognizer = DangerRecognizer()
    
    # 添加警戒区域
    recognizer.add_alert_region([(100, 100), (300, 100), (300, 300), (100, 300)], "禁区1")
    
    # 模拟特征
    class DummyFeature:
        def __init__(self, pos, end_pos, mag):
            self.position = pos
            self.end_position = end_pos
            self.magnitude = mag
    
    features = [
        DummyFeature((150, 150), (160, 160), 15),
        DummyFeature((200, 200), (210, 220), 20),
        DummyFeature((50, 50), (60, 70), 25),
    ]
    
    # 创建测试帧
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 处理帧
    alerts = recognizer.process_frame(frame, features)
    
    # 可视化
    vis_frame = recognizer.visualize(frame, alerts, features)
    
    # 显示结果
    cv2.imshow("Test", vis_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 