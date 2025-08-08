#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全功能视频监控系统 - 整合所有已开发模块的综合系统
结合了运动特征提取、危险行为识别、Web界面等功能
"""

import os
import sys
import time
import logging
import argparse
import cv2
import numpy as np
import threading
import json
from queue import Queue
from datetime import datetime
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("all_in_one_system.log")
    ]
)
logger = logging.getLogger("AllInOneSystem")

# 导入运动特征提取模块
try:
    from models.motion.motion_manager import MotionFeatureManager
    logger.info("成功导入MotionFeatureManager")
except ImportError as e:
    logger.error(f"导入MotionFeatureManager失败: {str(e)}")
    sys.exit(1)

# 导入危险行为识别模块
try:
    from danger_recognizer import DangerRecognizer
    logger.info("成功导入DangerRecognizer")
except ImportError as e:
    logger.error(f"导入DangerRecognizer失败: {str(e)}")
    sys.exit(1)

# 尝试导入Web界面模块
try:
    from flask import Flask, render_template, Response, jsonify, request
    import threading
    HAS_FLASK = True
    logger.info("成功导入Flask Web模块")
except ImportError:
    HAS_FLASK = False
    logger.warning("未找到Flask，Web界面功能将被禁用")

# 尝试导入AI模块
try:
    import torch
    from ultralytics import YOLO
    HAS_AI = True
    logger.info("成功导入AI功能模块")
except ImportError:
    HAS_AI = False
    logger.warning("未找到必要的AI依赖，AI功能将被禁用")

class AllInOneSystem:
    """全功能视频监控系统 - 整合所有模块"""
    
    def __init__(self, args):
        """初始化系统"""
        self.args = args
        
        # 创建输出目录
        os.makedirs(args.output, exist_ok=True)
        
        # 初始化状态
        self.running = False
        self.paused = False
        self.current_frame = None
        self.processed_frame = None
        self.last_frame_time = 0
        self.fps = 0
        self.frame_count = 0
        self.processed_count = 0
        self.alerts = []
        self.alert_count = 0
        self.start_time = time.time()
        
        # 线程和队列设置
        self.frame_queue = Queue(maxsize=30)
        self.result_queue = Queue(maxsize=30)
        self.threads = []
        
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
        
        # 初始化AI模块（如果可用）
        self.ai_model = None
        if args.enable_ai and HAS_AI:
            try:
                self.ai_model = YOLO(args.vision_model + ".pt")
                logger.info(f"成功加载AI模型: {args.vision_model}")
            except Exception as e:
                logger.error(f"加载AI模型失败: {str(e)}")
                self.ai_model = None
        
        # 初始化Web服务器（如果可用且启用）
        self.app = None
        if args.web_interface and HAS_FLASK:
            self.init_web_server()
        
        # 初始化视频录制器
        self.video_writer = None
        if args.record:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = os.path.join(args.output, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
            self.video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (args.width, args.height))
            logger.info(f"视频将录制到: {output_path}")
        
        logger.info("全功能视频监控系统初始化完成")
    
    def init_web_server(self):
        """初始化Web服务器"""
        self.app = Flask(__name__, template_folder='templates')
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/video_feed')
        def video_feed():
            """视频流路由"""
            return Response(self.generate_frames(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/stats')
        def stats():
            """统计信息API"""
            elapsed = time.time() - self.start_time
            return jsonify({
                'fps': self.fps,
                'frame_count': self.frame_count,
                'processed_count': self.processed_count,
                'alert_count': self.alert_count,
                'running_time': f"{elapsed:.1f}秒",
                'status': 'Running' if self.running else 'Stopped'
            })
        
        @self.app.route('/control', methods=['POST'])
        def control():
            """控制API"""
            action = request.json.get('action', '')
            if action == 'start':
                self.running = True
                return jsonify({'status': 'success', 'message': 'System started'})
            elif action == 'stop':
                self.running = False
                return jsonify({'status': 'success', 'message': 'System stopped'})
            elif action == 'pause':
                self.paused = not self.paused
                return jsonify({'status': 'success', 'message': f'System {"paused" if self.paused else "resumed"}'})
            return jsonify({'status': 'error', 'message': f'Unknown action: {action}'})
        
        def run_web_server():
            """在单独的线程中运行Web服务器"""
            self.app.run(host='0.0.0.0', port=self.args.web_port, debug=False, threaded=True)
        
        # 启动Web服务器线程
        web_thread = threading.Thread(target=run_web_server)
        web_thread.daemon = True
        web_thread.start()
        logger.info(f"Web服务器已启动，访问 http://localhost:{self.args.web_port}/")
    
    def generate_frames(self):
        """生成帧序列用于Web流"""
        while True:
            if self.processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', self.processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # 约30 FPS
    
    def start(self):
        """启动系统"""
        self.running = True
        
        # 启动捕获线程
        capture_thread = threading.Thread(target=self.capture_thread_func)
        capture_thread.daemon = True
        capture_thread.start()
        self.threads.append(capture_thread)
        
        # 启动处理线程
        process_thread = threading.Thread(target=self.process_thread_func)
        process_thread.daemon = True
        process_thread.start()
        self.threads.append(process_thread)
        
        logger.info("全功能视频监控系统已启动")
        
        # 如果没有Web界面，则使用显示循环
        if not (self.args.web_interface and HAS_FLASK):
            try:
                self.display_loop()
            except KeyboardInterrupt:
                logger.info("用户中断")
            finally:
                self.stop()
        else:
            # 使用Web界面时，主线程等待其他线程
            try:
                while self.running:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                logger.info("用户中断")
            finally:
                self.stop()
    
    def stop(self):
        """停止系统"""
        self.running = False
        
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        if self.video_writer is not None:
            self.video_writer.release()
        
        # 生成报告
        self.generate_report()
        
        logger.info("全功能视频监控系统已停止")
    
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
        
        # 设置摄像头参数（仅用于本地摄像头）
        if source == 0 or (isinstance(source, int) and source >= 0):
            # 尝试设置MJPG格式（如果支持）
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            # 尝试设置缓冲区大小最小
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 计算最大帧率
        max_fps = self.args.max_fps
        min_frame_time = 1.0 / max_fps
        
        frame_count = 0
        last_time = time.time()
        
        try:
            while self.running:
                # 处理暂停状态
                if self.paused:
                    time.sleep(0.1)
                    continue
                
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
                        if self.args.loop_video:
                            # 重新开始视频
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
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
                self.frame_count = frame_count
                self.current_frame = frame
                
                # 降低分辨率（如果启用）
                if self.args.process_scale < 1.0:
                    h, w = frame.shape[:2]
                    new_width = int(w * self.args.process_scale)
                    new_height = int(h * self.args.process_scale)
                    process_frame = cv2.resize(frame, (new_width, new_height))
                else:
                    process_frame = frame
                
                # 将帧放入队列
                if not self.frame_queue.full():
                    self.frame_queue.put((frame_count, process_frame, frame.copy() if process_frame is not frame else None, last_time))
                else:
                    # 如果队列满了，移除最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put((frame_count, process_frame, frame.copy() if process_frame is not frame else None, last_time))
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"视频捕获线程出错: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            cap.release()
            logger.info("视频捕获线程结束")
    
    def process_thread_func(self):
        """视频处理线程"""
        logger.info("开始视频处理线程")
        
        prev_frame = None
        processed_count = 0
        process_every = self.args.process_every  # 每N帧处理一次
        last_ai_frame = 0
        
        try:
            while self.running:
                # 处理暂停状态
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # 从队列中获取帧
                try:
                    frame_data = self.frame_queue.get(timeout=1.0)
                except:
                    continue
                
                frame_id, process_frame, original_frame, timestamp = frame_data
                
                # 仅处理每N帧
                if frame_id % process_every == 0:
                    processed_count += 1
                    self.processed_count = processed_count
                    
                    # 提取运动特征
                    features = self.motion_manager.extract_features(process_frame, prev_frame)
                    
                    # AI对象检测（如果启用）
                    object_detections = None
                    if self.ai_model is not None and (frame_id - last_ai_frame) >= self.args.ai_interval:
                        try:
                            results = self.ai_model(process_frame)
                            object_detections = self._parse_ai_results(results)
                            last_ai_frame = frame_id
                        except Exception as e:
                            logger.error(f"AI处理出错: {str(e)}")
                    
                    # 检测危险行为
                    alerts = self.danger_recognizer.process_frame(process_frame, features, object_detections)
                    
                    # 更新告警统计
                    if alerts:
                        self.alerts = alerts
                        self.alert_count += len(alerts)
                    
                    # 可视化结果
                    vis_frame = self.visualize_frame(original_frame or process_frame, process_frame, features, alerts, object_detections)
                    prev_frame = process_frame.copy()
                else:
                    # 对于跳过处理的帧，仍然要可视化，但不进行特征提取
                    vis_frame = self.visualize_frame(original_frame or process_frame, None, None, None, None)
                
                # 保存处理后的帧
                self.processed_frame = vis_frame
                
                # 计算FPS
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # 录制视频
                if self.video_writer is not None:
                    # 确保尺寸匹配
                    if vis_frame.shape[1] != self.args.width or vis_frame.shape[0] != self.args.height:
                        vis_frame_resized = cv2.resize(vis_frame, (self.args.width, self.args.height))
                        self.video_writer.write(vis_frame_resized)
                    else:
                        self.video_writer.write(vis_frame)
                
                # 清理帧队列以保持低延迟
                while self.frame_queue.qsize() > 5:
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        break
        
        except Exception as e:
            logger.error(f"视频处理线程出错: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            logger.info("视频处理线程结束")
    
    def _parse_ai_results(self, results):
        """解析AI检测结果"""
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf.item() > self.args.ai_confidence:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class': r.names[cls],
                        'confidence': conf
                    })
        return detections
    
    def visualize_frame(self, original_frame, process_frame=None, features=None, alerts=None, detections=None):
        """可视化处理结果"""
        if original_frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        vis_frame = original_frame.copy()
        
        # 如果启用简洁模式，则跳过复杂的可视化
        if self.args.minimal_ui:
            # 仅显示简单的状态信息
            self._add_minimal_info(vis_frame)
            return vis_frame
        
        # 可视化特征（如果有）
        if features and process_frame is not None:
            try:
                # 绘制特征
                vis_frame = self.motion_manager.visualize_features(vis_frame, features)
            except Exception as e:
                logger.error(f"可视化特征出错: {str(e)}")
        
        # 可视化AI检测结果（如果有）
        if detections:
            for det in detections:
                try:
                    x1, y1, x2, y2 = det['bbox']
                    cls = det['class']
                    conf = det['confidence']
                    
                    color = (0, 255, 0)  # 绿色
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis_frame, f"{cls} {conf:.2f}", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception as e:
                    logger.error(f"可视化检测结果出错: {str(e)}")
        
        # 可视化危险行为告警（如果有）
        if alerts:
            try:
                # 使用危险识别器的可视化功能
                vis_frame = self.danger_recognizer.visualize(vis_frame, alerts, features)
            except Exception as e:
                logger.error(f"可视化告警出错: {str(e)}")
        
        # 添加系统状态信息
        self._add_system_info(vis_frame)
        
        return vis_frame
    
    def _add_system_info(self, frame):
        """添加系统状态信息到帧"""
        h, w = frame.shape[:2]
        
        # 绘制系统信息背景
        cv2.rectangle(frame, (w - 240, 0), (w, 120), (0, 0, 0, 0.5), -1)
        
        # 显示基本信息
        info_items = [
            f"FPS: {self.fps:.1f}",
            f"帧: {self.frame_count}",
            f"处理: {self.processed_count}",
            f"告警: {self.alert_count}",
            f"运行时间: {time.time() - self.start_time:.0f}秒"
        ]
        
        for i, info in enumerate(info_items):
            cv2.putText(frame, info, (w - 230, 25 * (i + 1)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 显示当前模式
        mode_text = "模式: " 
        if self.args.enable_ai and self.ai_model is not None:
            mode_text += "AI+"
        mode_text += "运动"
        cv2.putText(frame, mode_text, (w - 230, 25 * (len(info_items) + 1)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
    
    def _add_minimal_info(self, frame):
        """添加最小化的系统信息到帧"""
        h, w = frame.shape[:2]
        
        # 仅在左上角显示FPS和告警数
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 25),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # 如果有告警，显示红色警告
        if self.alert_count > 0:
            cv2.putText(frame, f"告警: {self.alert_count}", (10, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    def display_loop(self):
        """显示循环"""
        logger.info("开始显示循环")
        
        try:
            while self.running:
                # 显示处理后的帧
                if self.processed_frame is not None:
                    cv2.imshow("全功能视频监控系统", self.processed_frame)
                
                # 检查键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Esc键退出
                    logger.info("用户按下Esc键，退出")
                    break
                elif key == ord('p'):  # 'p'键暂停/继续
                    self.paused = not self.paused
                    logger.info(f"系统{'暂停' if self.paused else '继续'}")
                elif key == ord('s'):  # 's'键保存当前帧
                    if self.processed_frame is not None:
                        save_path = os.path.join(self.args.output, f"frame_{self.frame_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                        cv2.imwrite(save_path, self.processed_frame)
                        logger.info(f"已保存当前帧: {save_path}")
                elif key == ord('r'):  # 'r'键重置统计
                    self.danger_recognizer.reset_stats()
                    self.alert_count = 0
                    logger.info("已重置统计信息")
                
                # 避免CPU占用过高
                time.sleep(0.01)
        
        except Exception as e:
            logger.error(f"显示循环出错: {str(e)}")
        finally:
            cv2.destroyAllWindows()
            logger.info("显示循环结束")
    
    def generate_report(self):
        """生成报告"""
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        processed_ratio = self.processed_count / max(1, self.frame_count) * 100
        
        report = "\n==== 系统报告 ====\n"
        report += f"运行时间: {elapsed:.2f} 秒\n"
        report += f"总帧数: {self.frame_count}\n"
        report += f"处理帧数: {self.processed_count} ({processed_ratio:.1f}%)\n"
        report += f"平均帧率: {avg_fps:.2f} FPS\n"
        report += f"告警总数: {self.alert_count}\n"
        
        # 添加告警分类统计
        alert_stats = self.danger_recognizer.get_alert_stats()
        report += "\n告警分类统计:\n"
        for alert_type, count in alert_stats.items():
            if count > 0:
                report += f"  - {alert_type}: {count}\n"
        
        # 添加系统配置信息
        report += "\n系统配置:\n"
        for arg, value in vars(self.args).items():
            report += f"  - {arg}: {value}\n"
        
        logger.info(report)
        
        # 保存报告到文件
        report_path = os.path.join(self.args.output, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已保存到: {report_path}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='全功能视频监控系统')
    
    # 输入参数
    parser.add_argument('--source', type=str, default='0', help='视频源 (0表示摄像头, 或者是视频文件路径)')
    parser.add_argument('--width', type=int, default=640, help='视频宽度')
    parser.add_argument('--height', type=int, default=480, help='视频高度')
    parser.add_argument('--loop_video', action='store_true', help='循环播放视频文件')
    
    # 处理参数
    parser.add_argument('--process_every', type=int, default=3, help='每N帧处理一次')
    parser.add_argument('--process_scale', type=float, default=1.0, help='处理分辨率缩放比例 (0.5=半分辨率)')
    parser.add_argument('--max_fps', type=int, default=30, help='最大帧率')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU加速')
    parser.add_argument('--use_motion_history', action='store_true', help='使用运动历史')
    parser.add_argument('--minimal_ui', action='store_true', help='使用最小化界面')
    
    # 危险行为检测参数
    parser.add_argument('--feature_threshold', type=int, default=80, help='特征点数量阈值')
    parser.add_argument('--area_threshold', type=float, default=0.05, help='运动区域阈值')
    parser.add_argument('--alert_cooldown', type=int, default=10, help='告警冷却帧数')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='最小置信度')
    parser.add_argument('--alert_region', type=str, help='警戒区域, 格式为坐标点列表, 例如: "[(100,100), (300,100), (300,300), (100,300)]"')
    
    # AI参数
    parser.add_argument('--enable_ai', action='store_true', help='启用AI功能')
    parser.add_argument('--vision_model', type=str, default='yolov8n', help='使用的视觉模型')
    parser.add_argument('--ai_interval', type=int, default=20, help='AI处理间隔帧数')
    parser.add_argument('--ai_confidence', type=float, default=0.4, help='AI检测置信度阈值')
    
    # Web界面参数
    parser.add_argument('--web_interface', action='store_true', help='启用Web界面')
    parser.add_argument('--web_port', type=int, default=5000, help='Web服务器端口')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='system_output', help='输出目录')
    parser.add_argument('--record', action='store_true', help='记录视频')
    parser.add_argument('--save_alerts', action='store_true', help='保存告警帧')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    system = AllInOneSystem(args)
    system.start()

if __name__ == "__main__":
    main() 