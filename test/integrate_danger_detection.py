#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
集成危险行为检测的视频监控系统 - 将危险行为检测模块集成到现有系统中
"""

import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np
import threading
from queue import Queue
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("danger_detection_system.log")
    ]
)
logger = logging.getLogger(__name__)

# 导入危险行为识别器
from danger_recognizer import DangerRecognizer

# 导入运动特征提取器
try:
    from models.motion.motion_manager import MotionFeatureManager
    logger.info("成功导入MotionFeatureManager")
except ImportError as e:
    logger.error(f"导入MotionFeatureManager失败: {str(e)}")
    sys.exit(1)

class VideoProcessor:
    """视频处理器 - 集成危险行为检测的视频处理系统"""
    
    def __init__(self, args):
        """初始化视频处理器"""
        self.args = args
        
        # 创建输出目录
        os.makedirs(args.output, exist_ok=True)
        
        # 初始化运动特征管理器
        self.motion_manager = MotionFeatureManager(
            use_optical_flow=True,
            use_motion_history=args.use_motion_history,
            optical_flow_method='farneback',
            use_gpu=args.use_gpu
        )
        
        # 初始化危险行为识别器
        danger_config = {
            'feature_count_threshold': args.feature_threshold,
            'motion_area_threshold': args.area_threshold,
            'alert_cooldown': args.alert_cooldown,
            'save_alerts': args.save_alerts,
            'alert_dir': os.path.join(args.output, 'alerts'),
            'min_confidence': args.min_confidence
        }
        self.danger_recognizer = DangerRecognizer(danger_config)
        
        # 如果指定了警戒区域，添加它
        if args.alert_region:
            try:
                regions = eval(args.alert_region)
                if isinstance(regions, list) and len(regions) >= 3:
                    self.danger_recognizer.add_alert_region(regions, "警戒区")
            except Exception as e:
                logger.error(f"解析警戒区域失败: {str(e)}")
        
        # 线程和队列
        self.frame_queue = Queue(maxsize=30)
        self.result_queue = Queue(maxsize=30)
        self.running = False
        self.capture_thread = None
        self.process_thread = None
        
        # 统计信息
        self.stats = {
            'start_time': time.time(),
            'total_frames': 0,
            'processed_frames': 0,
            'fps': 0,
            'alerts': 0
        }
    
    def start(self):
        """启动视频处理"""
        self.running = True
        
        # 启动捕获线程
        self.capture_thread = threading.Thread(target=self.capture_thread_func)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # 启动处理线程
        self.process_thread = threading.Thread(target=self.process_thread_func)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        logger.info("视频处理系统已启动")
        
        try:
            self.display_loop()
        except KeyboardInterrupt:
            logger.info("用户中断")
        finally:
            self.stop()
    
    def stop(self):
        """停止视频处理"""
        self.running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        
        # 生成最终报告
        self.generate_report()
        
        logger.info("视频处理系统已停止")
    
    def capture_thread_func(self):
        """视频捕获线程"""
        logger.info("开始视频捕获线程")
        
        # 打开视频源
        if self.args.source.isdigit():
            source = int(self.args.source)
        else:
            source = self.args.source
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            self.running = False
            return
        
        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        
        # 获取实际分辨率
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"视频分辨率: {actual_width}x{actual_height}")
        
        # 计算最大帧率
        max_fps = self.args.max_fps
        min_frame_time = 1.0 / max_fps
        
        frame_count = 0
        last_time = time.time()
        
        try:
            while self.running:
                # 限制帧率
                current_time = time.time()
                delta = current_time - last_time
                if delta < min_frame_time:
                    time.sleep(min_frame_time - delta)
                
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str) and not source.isdigit():
                        # 视频文件结束
                        logger.info("视频文件播放完毕")
                        self.running = False
                        break
                    else:
                        # 摄像头出错，尝试重新连接
                        logger.warning("视频帧获取失败，尝试重新连接...")
                        time.sleep(0.5)
                        cap.release()
                        cap = cv2.VideoCapture(source)
                        if not cap.isOpened():
                            logger.error("重新连接失败")
                            self.running = False
                            break
                        continue
                
                # 更新时间和计数
                last_time = time.time()
                frame_count += 1
                self.stats['total_frames'] = frame_count
                
                # 将帧放入队列
                if not self.frame_queue.full():
                    self.frame_queue.put((frame_count, frame.copy(), last_time))
                else:
                    # 如果队列满了，移除最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put((frame_count, frame.copy(), last_time))
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"视频捕获线程出错: {str(e)}")
        finally:
            cap.release()
            logger.info("视频捕获线程结束")
    
    def process_thread_func(self):
        """视频处理线程"""
        logger.info("开始视频处理线程")
        
        prev_frame = None
        processed_count = 0
        process_every = self.args.process_every  # 每N帧处理一次
        
        try:
            while self.running:
                # 从队列中获取帧
                try:
                    frame_data = self.frame_queue.get(timeout=1.0)
                except:
                    continue
                
                frame_id, frame, timestamp = frame_data
                
                # 只处理每N帧
                if frame_id % process_every == 0:
                    processed_count += 1
                    self.stats['processed_frames'] = processed_count
                    
                    # 提取运动特征
                    features = self.motion_manager.extract_features(frame, prev_frame)
                    
                    # 检测危险行为
                    alerts = self.danger_recognizer.process_frame(frame, features)
                    
                    # 更新告警统计
                    if alerts:
                        self.stats['alerts'] += len(alerts)
                    
                    # 可视化结果
                    vis_frame = self.danger_recognizer.visualize(frame, alerts, features)
                    prev_frame = frame.copy()
                else:
                    # 对于跳过处理的帧，仍然要可视化，但不进行特征提取
                    vis_frame = self.danger_recognizer.visualize(frame, None, None, show_debug=False)
                
                # 添加性能信息
                elapsed = time.time() - self.stats['start_time']
                fps = self.stats['total_frames'] / elapsed if elapsed > 0 else 0
                self.stats['fps'] = fps
                
                # 添加统计信息到帧
                cv2.putText(vis_frame, f"FPS: {fps:.1f}", (vis_frame.shape[1] - 150, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Frame: {frame_id}", (vis_frame.shape[1] - 150, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Alerts: {self.stats['alerts']}", (vis_frame.shape[1] - 150, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 放入结果队列
                if not self.result_queue.full():
                    self.result_queue.put((frame_id, vis_frame, alerts))
                else:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put((frame_id, vis_frame, alerts))
                    except:
                        pass
                
                # 清理帧队列以保持低延迟
                while self.frame_queue.qsize() > 5:
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        break
        
        except Exception as e:
            logger.error(f"视频处理线程出错: {str(e)}")
        finally:
            logger.info("视频处理线程结束")
    
    def display_loop(self):
        """显示循环"""
        logger.info("开始显示循环")
        
        video_writer = None
        if self.args.record:
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = os.path.join(self.args.output, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
            video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (self.args.width, self.args.height))
            logger.info(f"视频录制到: {output_path}")
        
        last_alert_save = time.time()
        
        try:
            while self.running:
                # 从结果队列获取处理后的帧
                try:
                    result = self.result_queue.get(timeout=1.0)
                except:
                    continue
                
                frame_id, vis_frame, alerts = result
                
                # 显示帧
                cv2.imshow("危险行为检测系统", vis_frame)
                
                # 录制视频
                if video_writer is not None:
                    video_writer.write(vis_frame)
                
                # 定期保存告警帧
                if alerts and (time.time() - last_alert_save) > 2.0:
                    alert_path = os.path.join(self.args.output, f"alert_{frame_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(alert_path, vis_frame)
                    last_alert_save = time.time()
                
                # 检查键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Esc键退出
                    logger.info("用户按下Esc键，退出")
                    break
                elif key == ord('s'):  # 's'键保存当前帧
                    save_path = os.path.join(self.args.output, f"frame_{frame_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(save_path, vis_frame)
                    logger.info(f"已保存当前帧: {save_path}")
                elif key == ord('r'):  # 'r'键重置统计
                    self.danger_recognizer.reset_stats()
                    self.stats['alerts'] = 0
                    logger.info("已重置统计信息")
        
        except Exception as e:
            logger.error(f"显示循环出错: {str(e)}")
        finally:
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            logger.info("显示循环结束")
    
    def generate_report(self):
        """生成报告"""
        elapsed = time.time() - self.stats['start_time']
        avg_fps = self.stats['total_frames'] / elapsed if elapsed > 0 else 0
        processed_ratio = self.stats['processed_frames'] / max(1, self.stats['total_frames']) * 100
        
        report = "\n==== 系统报告 ====\n"
        report += f"运行时间: {elapsed:.2f} 秒\n"
        report += f"总帧数: {self.stats['total_frames']}\n"
        report += f"处理帧数: {self.stats['processed_frames']} ({processed_ratio:.1f}%)\n"
        report += f"平均帧率: {avg_fps:.2f} FPS\n"
        report += f"告警总数: {self.stats['alerts']}\n"
        
        # 添加告警分类统计
        alert_stats = self.danger_recognizer.get_alert_stats()
        report += "\n告警分类统计:\n"
        for alert_type, count in alert_stats.items():
            if count > 0:
                report += f"  - {alert_type}: {count}\n"
        
        logger.info(report)
        
        # 保存报告到文件
        report_path = os.path.join(self.args.output, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已保存到: {report_path}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='危险行为检测系统')
    
    # 输入参数
    parser.add_argument('--source', type=str, default='0', help='视频源 (0表示摄像头, 或者是视频文件路径)')
    parser.add_argument('--width', type=int, default=640, help='视频宽度')
    parser.add_argument('--height', type=int, default=480, help='视频高度')
    
    # 处理参数
    parser.add_argument('--process_every', type=int, default=3, help='每N帧处理一次')
    parser.add_argument('--max_fps', type=int, default=30, help='最大帧率')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU加速')
    parser.add_argument('--use_motion_history', action='store_true', help='使用运动历史')
    
    # 检测参数
    parser.add_argument('--feature_threshold', type=int, default=100, help='特征点数量阈值')
    parser.add_argument('--area_threshold', type=float, default=0.05, help='运动区域阈值')
    parser.add_argument('--alert_cooldown', type=int, default=10, help='告警冷却帧数')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='最小置信度')
    parser.add_argument('--alert_region', type=str, help='警戒区域, 格式为坐标点列表, 例如: "[(100,100), (300,100), (300,300), (100,300)]"')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='danger_system_output', help='输出目录')
    parser.add_argument('--record', action='store_true', help='记录视频')
    parser.add_argument('--save_alerts', action='store_true', help='保存告警帧')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    processor = VideoProcessor(args)
    processor.start()

if __name__ == "__main__":
    main() 