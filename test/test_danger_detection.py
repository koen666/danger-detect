#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
危险动作检测验证脚本 - 用于测试和验证危险动作检测算法
"""

import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np
import threading
import traceback
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("danger_detection_test.log")
    ]
)
logger = logging.getLogger(__name__)

class SimpleBehaviorAnalyzer:
    """简单的行为分析器，用于检测危险行为"""
    
    def __init__(self):
        """初始化行为分析器"""
        self.history = []  # 存储历史特征
        self.history_length = 30  # 保存30帧的历史
        self.alert_threshold = {
            'sudden_motion': 10,  # 突然运动阈值，从40降低到10
            'large_area_motion': 0.05,  # 大面积运动阈值(占画面比例)，从0.25降低到0.05
            'fall_motion': 10,  # 摔倒特征阈值，从35降低到10
        }
        self.alert_cooldown = 5  # 告警冷却帧数，从20降低到5
        self.last_alert_frame = 0
        self.current_frame = 0
        
        # 警报统计
        self.alert_stats = {
            '突然剧烈运动': 0,
            '大范围异常运动': 0,
            '可能摔倒': 0,
            '异常移动模式': 0,
        }
        
        # 调试信息
        self.debug_info = {}
        
        # 判断条件的调试信息
        self.debug_thresholds = {}
    
    def update(self, features, frame_size):
        """更新特征历史并分析行为"""
        self.current_frame += 1
        
        # 提取当前帧的运动统计信息
        motion_stats = self._extract_motion_stats(features, frame_size)
        
        # 调试信息更新
        self.debug_info = {
            'feature_count': len(features) if features else 0,
            'avg_magnitude': motion_stats['avg_magnitude'],
            'max_magnitude': motion_stats['max_magnitude'],
            'motion_area': motion_stats['motion_area'],
            'vertical_motion': motion_stats['vertical_motion'],
            'direction_count': len(motion_stats['motion_directions']) 
        }
        
        # 添加到历史并保持固定长度
        self.history.append(motion_stats)
        if len(self.history) > self.history_length:
            self.history.pop(0)
        
        # 分析是否存在危险行为
        alerts = self._analyze_danger(features)
        
        # 更新统计信息
        for alert in alerts:
            if alert['type'] in self.alert_stats:
                self.alert_stats[alert['type']] += 1
        
        return alerts
    
    def _extract_motion_stats(self, features, frame_size):
        """从特征中提取运动统计信息"""
        frame_area = frame_size[0] * frame_size[1]
        stats = {
            'avg_magnitude': 0,
            'max_magnitude': 0,
            'motion_directions': {},  # 运动方向分布
            'motion_area': 0,  # 运动区域面积
            'vertical_motion': 0,  # 垂直运动分量
        }
        
        if not features:
            return stats
            
        # 计算各种统计量
        magnitudes = []
        motion_area = 0
        vertical_motion = 0
        
        for feature in features:
            if hasattr(feature, 'magnitude'):
                magnitudes.append(feature.magnitude)
            if hasattr(feature, 'position') and hasattr(feature, 'end_position'):
                # 计算运动方向
                dx = feature.end_position[0] - feature.position[0]
                dy = feature.end_position[1] - feature.position[1]
                
                # 垂直运动（正值表示向下，负值表示向上）
                vertical_motion += dy
                
                # 运动方向量化为8个方向
                angle = np.degrees(np.arctan2(dy, dx))
                direction = round(angle / 45) * 45
                if direction in stats['motion_directions']:
                    stats['motion_directions'][direction] += 1
                else:
                    stats['motion_directions'][direction] = 1
            
            # 近似计算运动区域
            if hasattr(feature, 'magnitude'):
                motion_area += feature.magnitude * 10  # 简化计算
        
        if magnitudes:
            stats['avg_magnitude'] = sum(magnitudes) / len(magnitudes)
            stats['max_magnitude'] = max(magnitudes)
        
        stats['motion_area'] = min(1.0, motion_area / frame_area)  # 归一化为占比
        stats['vertical_motion'] = vertical_motion / max(1, len(features))
        
        return stats
    
    def _analyze_danger(self, features):
        """分析是否存在危险行为"""
        if len(self.history) < 3:  # 需要足够的历史数据
            return []
        
        alerts = []
        self.debug_thresholds = {}  # 清空上一帧的调试信息
        
        # 特征数量检测 - 直接基于特征点数量判断
        feature_count = len(features) if features else 0
        self.debug_thresholds['feature_count'] = feature_count
        self.debug_thresholds['feature_threshold'] = 100  # 设定特征点数量阈值
        
        # 基于特征点数量的剧烈运动检测
        if feature_count > 100:
            # 只要有足够多的特征点，就认为可能有剧烈运动
            if self.current_frame - self.last_alert_frame > self.alert_cooldown:
                alerts.append({
                    'type': '突然剧烈运动',
                    'confidence': min(1.0, feature_count / 300),
                    'frame': self.current_frame
                })
                self.last_alert_frame = self.current_frame
        
        # 如果有历史数据，比较当前帧和前几帧的特征数量变化
        if len(self.history) >= 5:
            # 计算前几帧的平均特征数量
            prev_features = 0
            for i in range(-5, -1):
                if i < len(self.history):
                    prev_features += self.debug_info.get('feature_count', 0)
            
            prev_features = prev_features / 4 if prev_features > 0 else 0.1
            feature_change_ratio = feature_count / max(1, prev_features)
            
            self.debug_thresholds['feature_change_ratio'] = feature_change_ratio
            
            # 特征数量突然变化检测
            if feature_change_ratio > 1.5 and feature_count > 50:
                if self.current_frame - self.last_alert_frame > self.alert_cooldown:
                    alerts.append({
                        'type': '突然变化',
                        'confidence': min(1.0, feature_change_ratio / 3),
                        'frame': self.current_frame
                    })
                    self.last_alert_frame = self.current_frame
        
        # 突然运动检测 - 比较当前帧与前几帧的运动幅度变化
        current = self.history[-1]['avg_magnitude']
        prev_avg = sum(h['avg_magnitude'] for h in self.history[-6:-1]) / 5 if len(self.history) >= 6 else 0.1
        
        # 保存调试数据
        self.debug_thresholds['current_magnitude'] = current
        self.debug_thresholds['prev_avg_magnitude'] = prev_avg
        self.debug_thresholds['magnitude_ratio'] = current / max(0.1, prev_avg)
        self.debug_thresholds['sudden_motion_threshold'] = self.alert_threshold['sudden_motion']
        
        # 降低倍数要求，从1.5降低到1.2
        if current > prev_avg * 1.2 and current > self.alert_threshold['sudden_motion']:
            if self.current_frame - self.last_alert_frame > self.alert_cooldown:
                alerts.append({
                    'type': '突然剧烈运动',
                    'confidence': min(1.0, current / (prev_avg * 3)),
                    'frame': self.current_frame
                })
                self.last_alert_frame = self.current_frame
        
        # 大面积运动检测
        motion_area = self.history[-1]['motion_area']
        self.debug_thresholds['motion_area'] = motion_area
        self.debug_thresholds['area_threshold'] = self.alert_threshold['large_area_motion']
        
        if motion_area > self.alert_threshold['large_area_motion']:
            if self.current_frame - self.last_alert_frame > self.alert_cooldown:
                alerts.append({
                    'type': '大范围异常运动',
                    'confidence': min(1.0, motion_area / self.alert_threshold['large_area_motion']),
                    'frame': self.current_frame
                })
                self.last_alert_frame = self.current_frame
        
        # 摔倒检测 - 检测垂直向下的快速运动后突然停止
        vertical_last_frames = sum(h['vertical_motion'] for h in self.history[-10:-5])
        vertical_recent = abs(sum(h['vertical_motion'] for h in self.history[-3:]))
        
        self.debug_thresholds['vertical_motion'] = vertical_last_frames
        self.debug_thresholds['vertical_recent'] = vertical_recent
        self.debug_thresholds['fall_threshold'] = self.alert_threshold['fall_motion']
        
        if (len(self.history) >= 10 and 
            vertical_last_frames > self.alert_threshold['fall_motion'] and
            vertical_recent < 10):
            
            if self.current_frame - self.last_alert_frame > self.alert_cooldown:
                alerts.append({
                    'type': '可能摔倒',
                    'confidence': 0.7,
                    'frame': self.current_frame
                })
                self.last_alert_frame = self.current_frame
        
        # 检测异常移动模式（如徘徊）
        if len(self.history) >= 20:
            direction_changes = 0
            prev_direction = None
            
            # 计算方向变化次数
            for i in range(-20, 0):
                main_direction = None
                max_count = 0
                
                for direction, count in self.history[i]['motion_directions'].items():
                    if count > max_count:
                        max_count = count
                        main_direction = direction
                
                if main_direction is not None and prev_direction is not None:
                    if abs(main_direction - prev_direction) >= 90:
                        direction_changes += 1
                
                prev_direction = main_direction
            
            # 如果方向变化频繁，可能是异常行为
            if direction_changes >= 8 and self.current_frame - self.last_alert_frame > self.alert_cooldown:
                alerts.append({
                    'type': '异常移动模式',
                    'confidence': min(1.0, direction_changes / 12),
                    'frame': self.current_frame
                })
                self.last_alert_frame = self.current_frame
        
        return alerts

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='危险动作检测验证脚本')
    
    # 输入源参数
    parser.add_argument('--source', type=str, default='0', help='视频源 (0表示摄像头, 或者是视频文件路径)')
    parser.add_argument('--width', type=int, default=640, help='视频宽度')
    parser.add_argument('--height', type=int, default=480, help='视频高度')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='danger_test_output', help='输出目录')
    parser.add_argument('--record', action='store_true', help='记录视频')
    parser.add_argument('--save_alerts', action='store_true', help='保存告警帧')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # 导入Motion Manager
        from models.motion.motion_manager import MotionFeatureManager
        logger.info("成功导入MotionFeatureManager")
    except ImportError as e:
        logger.error(f"导入MotionFeatureManager失败: {str(e)}")
        return
    
    # 初始化运动特征管理器
    motion_manager = MotionFeatureManager(
        use_optical_flow=True,
        use_motion_history=False,
        optical_flow_method='farneback',
        use_gpu=False
    )
    
    # 初始化行为分析器
    behavior_analyzer = SimpleBehaviorAnalyzer()
    
    # 打开视频源
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"无法打开视频源: {source}")
        return
    
    # 设置视频分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # 准备视频写入器
    video_writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = os.path.join(args.output, f"test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
        video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (args.width, args.height))
        logger.info(f"视频将保存到: {output_path}")
    
    # 初始化变量
    frame_count = 0
    prev_frame = None
    start_time = time.time()
    alert_frames = []
    
    logger.info("开始测试危险动作检测...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and not source.isdigit():
                    # 视频文件播放结束
                    break
                else:
                    logger.error("无法读取视频帧")
                    continue
            
            frame_count += 1
            
            # 提取运动特征
            features = motion_manager.extract_features(frame, prev_frame)
            
            # 分析危险行为
            alerts = behavior_analyzer.update(features, (frame.shape[1], frame.shape[0]))
            
            # 可视化显示
            vis_frame = frame.copy()
            
            # 显示特征
            if features:
                vis_frame = motion_manager.visualize_features(vis_frame, features)
            
            # 显示调试信息
            y_offset = 30
            for key, value in behavior_analyzer.debug_info.items():
                text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
                cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            # 显示阈值判断的调试信息
            for key, value in behavior_analyzer.debug_thresholds.items():
                text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
                cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
                y_offset += 20
            
            # 显示告警
            if alerts:
                # 记录告警帧
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                alert_info = {
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'alerts': alerts,
                    'image': vis_frame.copy()
                }
                alert_frames.append(alert_info)
                
                # 显示告警信息
                for i, alert in enumerate(alerts):
                    alert_text = f"{alert['type']} ({alert['confidence']:.2f})"
                    cv2.putText(vis_frame, alert_text, (10, frame.shape[0] - 30 - i*30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 在画面边缘添加红色告警框
                    cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1], vis_frame.shape[0]), (0, 0, 255), 5)
            
            # 显示基本信息
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (vis_frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(vis_frame, f"Frame: {frame_count}", (vis_frame.shape[1] - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 显示告警统计
            cv2.putText(vis_frame, "告警统计:", (vis_frame.shape[1] - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos = 100
            for alert_type, count in behavior_analyzer.alert_stats.items():
                if count > 0:
                    cv2.putText(vis_frame, f"{alert_type}: {count}", (vis_frame.shape[1] - 120, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    y_pos += 20
            
            # 显示图像
            cv2.imshow('危险动作检测测试', vis_frame)
            
            # 录制视频
            if video_writer is not None:
                video_writer.write(vis_frame)
            
            # 保存告警帧
            if args.save_alerts and alerts:
                for alert in alerts:
                    alert_path = os.path.join(args.output, f"alert_{alert['type']}_{frame_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(alert_path, vis_frame)
                    logger.info(f"已保存告警帧: {alert_path}")
            
            # 更新上一帧
            prev_frame = frame.copy()
            
            # 检查退出
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc键退出
                break
            elif key == ord('s'):  # 's'键保存当前帧
                save_path = os.path.join(args.output, f"frame_{frame_count}.jpg")
                cv2.imwrite(save_path, vis_frame)
                logger.info(f"已保存当前帧: {save_path}")
    
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # 释放资源
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # 生成测试报告
        logger.info("\n==== 测试报告 ====")
        logger.info(f"总帧数: {frame_count}")
        logger.info(f"测试时长: {time.time() - start_time:.2f} 秒")
        logger.info(f"告警总数: {sum(behavior_analyzer.alert_stats.values())}")
        
        for alert_type, count in behavior_analyzer.alert_stats.items():
            if count > 0:
                logger.info(f"  - {alert_type}: {count} 次")
        
        # 保存告警记录
        if alert_frames and args.save_alerts:
            alert_log_path = os.path.join(args.output, f"alert_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(alert_log_path, 'w', encoding='utf-8') as f:
                f.write("==== 告警记录 ====\n")
                for alert_info in alert_frames:
                    f.write(f"帧: {alert_info['frame']}, 时间: {alert_info['timestamp']}\n")
                    for alert in alert_info['alerts']:
                        f.write(f"  - {alert['type']} (置信度: {alert['confidence']:.2f})\n")
                    f.write("\n")
            
            logger.info(f"告警记录已保存到: {alert_log_path}")

if __name__ == "__main__":
    main() 