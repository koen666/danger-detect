#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
import os
import time
from datetime import datetime
import logging

from models.video_capture import VideoCaptureManager
from models.frame_processor import FrameProcessor
from models.object_detection import ObjectDetector
from models.behavior_analysis import BehaviorAnalysisSystem

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BehaviorAnalysisTest')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试行为分析系统')
    
    # 视频源参数
    parser.add_argument('--source', type=str, default='0', 
                      help='视频源，可以是摄像头索引(0)或视频文件路径')
    parser.add_argument('--width', type=int, default=640,
                      help='视频帧宽度')
    parser.add_argument('--height', type=int, default=480,
                      help='视频帧高度')
    
    # 物体检测参数
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                      help='物体检测模型路径')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='物体检测置信度阈值')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                      help='只检测特定类别，例如 --classes 0 1 2 表示只检测第0、1、2类')
    
    # 行为分析参数
    parser.add_argument('--use_flow', action='store_true',
                      help='使用光流分析')
    parser.add_argument('--use_motion', action='store_true',
                      help='使用运动历史')
    parser.add_argument('--history', type=int, default=60,
                      help='轨迹历史最大帧数')
    
    # 显示参数
    parser.add_argument('--show_detections', action='store_true',
                      help='显示物体检测结果')
    parser.add_argument('--show_motion', action='store_true',
                      help='显示运动特征')
    parser.add_argument('--show_behaviors', action='store_true',
                      help='显示行为分析结果')
    parser.add_argument('--show_heatmap', action='store_true',
                      help='显示运动热图')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='输出目录')
    parser.add_argument('--save_frames', action='store_true',
                      help='保存关键帧')
    
    args = parser.parse_args()
    
    # 默认显示所有可视化
    if not any([args.show_detections, args.show_motion, args.show_behaviors, args.show_heatmap]):
        args.show_detections = True
        args.show_motion = True
        args.show_behaviors = True
        args.show_heatmap = True
    
    # 默认使用所有分析方法
    if not any([args.use_flow, args.use_motion]):
        args.use_flow = True
        args.use_motion = True
    
    return args

class BehaviorAnalysisDemo:
    """行为分析演示类"""
    
    def __init__(self, args):
        """初始化演示类"""
        self.args = args
        self.logger = logging.getLogger('BehaviorAnalysisDemo')
        
        # 输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 初始化视频捕获器
        self.capture = VideoCaptureManager(
            source=args.source,
            width=args.width,
            height=args.height
        )
        
        # 初始化物体检测器
        self.detector = ObjectDetector(
            model_path=args.model,
            confidence_threshold=args.conf,
            classes=args.classes
        )
        
        # 初始化帧处理器
        self.frame_processor = FrameProcessor()
        
        # 初始化行为分析系统
        self.behavior_system = BehaviorAnalysisSystem(
            frame_width=args.width,
            frame_height=args.height,
            use_optical_flow=args.use_flow,
            use_motion_history=args.use_motion,
            max_trajectory_history=args.history
        )
        
        # 状态变量
        self.running = True
        self.paused = False
        self.frame_count = 0
        self.last_saved = 0
        self.show_help = True
        self.show_fps = True
        
        # 性能统计
        self.processing_times = []
        self.detection_times = []
        self.analysis_times = []
        self.max_times = 30  # 保存最近30帧的处理时间
        
        # 提示文本
        self.help_text = [
            "按键控制:",
            "ESC/Q: 退出程序",
            "空格: 暂停/继续",
            "H: 显示/隐藏帮助",
            "F: 显示/隐藏FPS",
            "D: 显示/隐藏物体检测",
            "M: 显示/隐藏运动特征",
            "B: 显示/隐藏行为分析",
            "T: 显示/隐藏热图",
            "S: 保存当前帧和分析结果",
            "R: 重置行为分析系统"
        ]
        
        self.logger.info("行为分析演示初始化完成")
    
    def connect(self):
        """连接视频源"""
        if not self.capture.connect():
            self.logger.error(f"无法连接视频源: {self.args.source}")
            return False
        
        self.logger.info(f"成功连接视频源: {self.args.source}")
        return True
    
    def process_frame(self, frame):
        """处理帧并进行物体检测和行为分析"""
        if frame is None:
            return None, None, 0.0
        
        # 1. 预处理帧
        processed_frame = self.frame_processor.process(frame)
        
        # 2. 物体检测
        det_start = time.time()
        tracks = self.detector.detect_and_track(processed_frame)
        det_time = time.time() - det_start
        
        # 3. 行为分析
        analysis_start = time.time()
        analysis_results = self.behavior_system.process_frame(processed_frame, tracks)
        analysis_time = time.time() - analysis_start
        
        # 更新统计
        self.detection_times.append(det_time)
        self.analysis_times.append(analysis_time)
        
        if len(self.detection_times) > self.max_times:
            self.detection_times.pop(0)
        if len(self.analysis_times) > self.max_times:
            self.analysis_times.pop(0)
        
        return processed_frame, tracks, analysis_results
    
    def visualize_results(self, frame, tracks, analysis_results):
        """可视化处理结果"""
        output = frame.copy()
        
        # 1. 绘制物体检测结果
        if self.args.show_detections and tracks:
            output = self.detector.draw_tracks(output, tracks)
        
        # 2. 创建综合可视化
        output = self.behavior_system.visualize_combined_results(
            output,
            include_motion=self.args.show_motion,
            include_behavior=self.args.show_behaviors,
            include_heatmap=self.args.show_heatmap,
            heatmap_alpha=0.3
        )
        
        # 3. 添加状态信息
        if self.show_fps and self.processing_times:
            fps = 1.0 / (sum(self.processing_times) / len(self.processing_times))
            det_time = sum(self.detection_times) / len(self.detection_times) * 1000
            analysis_time = sum(self.analysis_times) / len(self.analysis_times) * 1000
            
            cv2.putText(
                output,
                f"FPS: {fps:.1f} | Det: {det_time:.1f}ms | Ana: {analysis_time:.1f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # 4. 添加暂停状态
        if self.paused:
            cv2.putText(
                output,
                "已暂停",
                (output.shape[1] // 2 - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
        
        # 5. 添加帮助信息
        if self.show_help:
            self.add_help_overlay(output)
        
        return output
    
    def add_help_overlay(self, frame):
        """添加帮助信息叠加层"""
        overlay = frame.copy()
        
        # 半透明背景
        cv2.rectangle(
            overlay,
            (10, 50),
            (300, 50 + 25 * len(self.help_text)),
            (0, 0, 0),
            -1
        )
        
        # 设置透明度
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 添加文本
        for i, line in enumerate(self.help_text):
            cv2.putText(
                frame,
                line,
                (15, 75 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
    
    def save_results(self, frame, analysis_results):
        """保存结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存当前帧
        frame_path = os.path.join(self.args.output_dir, f"frame_{timestamp}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # 2. 保存行为分析结果
        self.behavior_system.save_analysis_results(
            self.args.output_dir,
            f"analysis_{timestamp}"
        )
        
        # 3. 保存运动热图
        heatmap = self.behavior_system.create_motion_heatmap()
        if heatmap is not None:
            heatmap_path = os.path.join(self.args.output_dir, f"heatmap_{timestamp}.jpg")
            cv2.imwrite(heatmap_path, heatmap)
        
        # 4. 保存行为统计
        summary = self.behavior_system.generate_behavior_summary()
        summary_path = os.path.join(self.args.output_dir, f"summary_{timestamp}.json")
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"结果已保存到目录: {self.args.output_dir}")
        
        return True
    
    def handle_key(self, key):
        """处理键盘输入"""
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC 或 Q
            self.running = False
        elif key == 32:  # 空格
            self.paused = not self.paused
        elif key == ord('h') or key == ord('H'):  # H
            self.show_help = not self.show_help
        elif key == ord('f') or key == ord('F'):  # F
            self.show_fps = not self.show_fps
        elif key == ord('d') or key == ord('D'):  # D
            self.args.show_detections = not self.args.show_detections
        elif key == ord('m') or key == ord('M'):  # M
            self.args.show_motion = not self.args.show_motion
        elif key == ord('b') or key == ord('B'):  # B
            self.args.show_behaviors = not self.args.show_behaviors
        elif key == ord('t') or key == ord('T'):  # T
            self.args.show_heatmap = not self.args.show_heatmap
        elif key == ord('s') or key == ord('S'):  # S
            return "save"
        elif key == ord('r') or key == ord('R'):  # R
            self.behavior_system.reset()
            self.logger.info("行为分析系统已重置")
        
        return None
    
    def run(self):
        """运行演示"""
        if not self.connect():
            return False
        
        self.logger.info("开始演示, 按 'Q' 或 'ESC' 退出")
        
        try:
            while self.running:
                start_time = time.time()
                
                # 获取帧
                if not self.paused:
                    ret, frame = self.capture.read()
                    if not ret or frame is None:
                        self.logger.warning("无法获取帧，退出")
                        break
                    
                    # 处理帧
                    processed_frame, tracks, analysis_results = self.process_frame(frame)
                    self.frame_count += 1
                
                # 可视化结果
                output = self.visualize_results(processed_frame, tracks, analysis_results)
                
                # 显示结果
                cv2.imshow("行为分析系统", output)
                
                # 计算处理时间
                process_time = time.time() - start_time
                self.processing_times.append(process_time)
                if len(self.processing_times) > self.max_times:
                    self.processing_times.pop(0)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                action = self.handle_key(key)
                
                # 执行操作
                if action == "save":
                    self.save_results(processed_frame, analysis_results)
                
                # 保持大约30FPS的显示
                if not self.paused:
                    time_to_wait = max(1, int((1.0/30 - process_time) * 1000))
                    if time_to_wait > 0:
                        cv2.waitKey(time_to_wait)
            
            # 生成并显示最终行为分析统计
            summary = self.behavior_system.generate_behavior_summary()
            self.logger.info(f"行为分析统计: {len(summary['behaviors'])} 种行为类型, {summary['total_count']} 个行为事件")
            
            # 保存最终结果
            self.save_results(processed_frame, analysis_results)
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("用户中断，退出")
            return True
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.capture.release()
        cv2.destroyAllWindows()
        self.logger.info("资源已释放")

def main():
    """主函数"""
    args = parse_args()
    demo = BehaviorAnalysisDemo(args)
    demo.run()

if __name__ == "__main__":
    main() 