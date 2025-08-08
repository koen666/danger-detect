#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频运动分析系统 - Web版
使用Flask提供Web界面，显示视频运动分析结果
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
from flask import Flask, Response, render_template, request, jsonify

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("motion_web_system.log")
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
frame_buffer = None
features_count = 0
feature_types = {}
fps = 0
frame_count = 0
running = False
processing_active = False  # 新增变量，用于控制处理状态
lock = threading.Lock()
processing_thread = None  # 新增变量，用于存储处理线程引用
test_mode = False

# 创建Flask应用
app = Flask(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='视频运动分析系统 - Web版')
    
    # 视频源参数
    parser.add_argument('--source', type=str, default='0', help='视频源 (0表示摄像头, 或者是视频文件路径)')
    parser.add_argument('--width', type=int, default=640, help='视频宽度')
    parser.add_argument('--height', type=int, default=480, help='视频高度')
    
    # 运动特征参数
    parser.add_argument('--use_optical_flow', action='store_true', help='启用光流特征提取')
    parser.add_argument('--use_motion_history', action='store_true', help='启用运动历史特征提取')
    parser.add_argument('--optical_flow_method', type=str, default='farneback', 
                       choices=['farneback', 'lucas_kanade'], help='光流算法')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU加速（如果可用）')
    
    # Web服务器参数
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web服务器主机')
    parser.add_argument('--port', type=int, default=5000, help='Web服务器端口')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--save_frames', action='store_true', help='保存处理后的帧')
    parser.add_argument('--save_interval', type=int, default=30, help='保存帧的间隔')
    
    return parser.parse_args()

def video_processing_thread(args):
    """视频处理线程"""
    global frame_buffer, features_count, feature_types, fps, frame_count, running, processing_active
    
    try:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 导入运动特征管理器
        try:
            from models.motion.motion_manager import MotionFeatureManager
            logger.info("成功导入MotionFeatureManager")
        except ImportError as e:
            logger.error(f"导入MotionFeatureManager失败: {str(e)}")
            return
        
        # 初始化运动特征管理器
        motion_manager = MotionFeatureManager(
            use_optical_flow=args.use_optical_flow,
            use_motion_history=args.use_motion_history,
            optical_flow_method=args.optical_flow_method,
            use_gpu=args.use_gpu
        )
        
        # 打开视频源
        if args.source.isdigit():
            source = int(args.source)
        else:
            source = args.source
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            # 使用测试模式
            logger.info("切换到测试模式")
            frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            test_mode = True
        else:
            # 设置视频分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            # 设置缓冲区大小为1以减少延迟
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = cap.read()
            if not ret:
                logger.error("无法读取视频帧")
                return
            test_mode = False
            logger.info(f"成功打开视频源: {source}")
        
        # 初始化状态变量
        start_time = time.time()
        prev_frame = None
        local_frame_count = 0
        skip_count = 0  # 帧跳过计数器
        max_fps = 15  # 最大处理帧率，防止过度占用CPU
        
        logger.info("开始处理视频流...")
        running = True
        processing_active = True
        
        while running and processing_active:
            # 获取帧
            if test_mode:
                # 在测试模式下生成移动的圆
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                center_x = int(args.width/2 + args.width/4 * np.sin(local_frame_count / 30.0))
                center_y = int(args.height/2 + args.height/4 * np.cos(local_frame_count / 30.0))
                cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
            else:
                # 尝试读取最新帧，丢弃缓冲区中的旧帧
                for _ in range(3):  # 丢弃一些帧以获取最新的
                    ret, temp_frame = cap.read()
                    if not ret:
                        break
                
                if not ret:
                    # 对于视频文件，可以循环播放
                    if isinstance(source, str) and not source.isdigit():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                frame = temp_frame
            
            # 控制处理帧率
            current_time = time.time()
            elapsed = current_time - start_time
            current_fps = local_frame_count / elapsed if elapsed > 0 else 0
            
            # 如果当前帧率超过目标，则跳过此帧的处理
            if current_fps > max_fps:
                skip_count += 1
                if skip_count % 5 == 0:  # 每5帧仍处理一帧，防止完全不更新
                    pass
                else:
                    # 简单更新帧缓冲区但不做特征提取
                    with lock:
                        frame_buffer = frame.copy()
                    time.sleep(0.01)  # 短暂休眠减轻CPU负担
                    continue
            
            # 降低分辨率进行处理以提高性能
            process_height = min(frame.shape[0], 240)
            process_width = int(process_height * frame.shape[1] / frame.shape[0])
            process_frame = cv2.resize(frame, (process_width, process_height))
            
            if prev_frame is None:
                process_prev_frame = None
            else:
                process_prev_frame = cv2.resize(prev_frame, (process_width, process_height))
            
            # 提取运动特征
            features = motion_manager.extract_features(process_frame, process_prev_frame)
            
            # 可视化特征
            vis_frame = frame.copy()
            if features:
                # 将特征坐标调整回原始分辨率
                scale_x = frame.shape[1] / process_width
                scale_y = frame.shape[0] / process_height
                for feature in features:
                    if hasattr(feature, 'position'):
                        feature.position = (int(feature.position[0] * scale_x), 
                                          int(feature.position[1] * scale_y))
                    if hasattr(feature, 'magnitude'):
                        feature.magnitude = feature.magnitude * scale_x  # 缩放幅度
                
                vis_frame = motion_manager.visualize_features(vis_frame, features)
            
            # 获取运动历史图像
            if args.use_motion_history:
                mhi = motion_manager.get_motion_history_image()
                if mhi is not None:
                    try:
                        # 确保MHI是8位格式
                        if mhi.dtype != np.uint8:
                            mhi_norm = cv2.normalize(mhi, None, 0, 255, cv2.NORM_MINMAX)
                            mhi_8bit = np.uint8(mhi_norm)
                        else:
                            mhi_8bit = mhi
                        
                        # 转换为彩色显示
                        mhi_color = cv2.applyColorMap(mhi_8bit, cv2.COLORMAP_JET)
                        # 调整大小
                        mhi_color = cv2.resize(mhi_color, (vis_frame.shape[1] // 4, vis_frame.shape[0] // 4))
                        # 放在右上角
                        vis_frame[10:10+mhi_color.shape[0], vis_frame.shape[1]-mhi_color.shape[1]-10:vis_frame.shape[1]-10] = mhi_color
                    except Exception as e:
                        logger.error(f"可视化运动历史图像时出错: {str(e)}")
            
            # 统计特征类型
            local_feature_types = {}
            for feature in features:
                feature_type = feature.type
                if feature_type in local_feature_types:
                    local_feature_types[feature_type] += 1
                else:
                    local_feature_types[feature_type] = 1
            
            # 显示帧信息和特征统计
            elapsed_time = time.time() - start_time
            local_fps = local_frame_count / elapsed_time if elapsed_time > 0 else 0
            
            cv2.putText(vis_frame, f"FPS: {local_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Frame: {local_frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Features: {len(features)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            y_pos = 120
            for feature_type, count in local_feature_types.items():
                cv2.putText(vis_frame, f"{feature_type}: {count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
            
            # 显示启用的功能
            status_text = []
            if args.use_optical_flow:
                status_text.append("光流:开")
            if args.use_motion_history:
                status_text.append("运动历史:开")
            
            # 显示处理状态
            if processing_active:
                status_text.append("状态:运行中")
            else:
                status_text.append("状态:已暂停")
            
            status_str = " | ".join(status_text)
            cv2.putText(vis_frame, status_str, (10, vis_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 保存结果
            if args.save_frames and local_frame_count % args.save_interval == 0:
                output_path = os.path.join(args.output_dir, f"frame_{local_frame_count:05d}.jpg")
                cv2.imwrite(output_path, vis_frame)
                logger.info(f"保存帧到: {output_path}")
            
            # 更新全局变量
            with lock:
                frame_buffer = vis_frame.copy()
                features_count = len(features)
                feature_types = local_feature_types.copy()
                fps = local_fps
                frame_count = local_frame_count
            
            # 更新状态
            prev_frame = frame.copy()
            local_frame_count += 1
            
            # 适当降低帧率，减轻CPU负担
            frame_time = time.time() - current_time
            sleep_time = max(0, 1.0/max_fps - frame_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # 清理资源
        if not test_mode and cap.isOpened():
            cap.release()
        
        # 显示统计信息
        elapsed = time.time() - start_time
        logger.info(f"处理完成: {local_frame_count} 帧在 {elapsed:.2f} 秒内 (平均 {local_frame_count/elapsed:.2f} FPS)")
    
    except Exception as e:
        logger.error(f"视频处理线程出错: {str(e)}")
        logger.error(traceback.format_exc())
        running = False
        processing_active = False

def generate_frames():
    """生成视频帧流"""
    global frame_buffer
    
    while running:
        if frame_buffer is not None:
            with lock:
                frame = frame_buffer.copy()
            
            # 转换为JPEG格式
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # 返回视频流数据
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 适当降低帧率，减轻CPU负担
        time.sleep(0.03)  # 约30 FPS

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流端点"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """获取统计信息"""
    global features_count, feature_types, fps, frame_count
    
    with lock:
        response = {
            'features_count': features_count,
            'feature_types': feature_types,
            'fps': fps,
            'frame_count': frame_count
        }
    
    return jsonify(response)

@app.route('/start', methods=['POST'])
def start_processing():
    """开始处理"""
    global processing_active, processing_thread, running
    
    logger.info("收到开始处理请求")
    with lock:
        if not processing_active:
            processing_active = True
            logger.info("启动视频处理...")
            # 重启线程如果需要
            if processing_thread is None or not processing_thread.is_alive():
                args = parse_args()
                processing_thread = threading.Thread(target=video_processing_thread, args=(args,))
                processing_thread.daemon = True
                processing_thread.start()
                logger.info("创建并启动了新的处理线程")
            
    return jsonify({'status': 'success', 'message': '处理已启动', 'active': processing_active})

@app.route('/stop', methods=['POST'])
def stop_processing():
    """停止处理"""
    global processing_active
    
    logger.info("收到停止处理请求")
    with lock:
        if processing_active:
            processing_active = False
            logger.info("停止视频处理...")
    
    return jsonify({'status': 'success', 'message': '处理已停止', 'active': processing_active})

def main():
    """主函数"""
    global processing_thread
    
    args = parse_args()
    
    # 创建模板目录
    os.makedirs('templates', exist_ok=True)
    
    # 创建HTML模板
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>24小时视频运动分析系统</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
            background-color: #f8f9fa;
        }
        .video-container {
            margin-top: 20px;
            text-align: center;
        }
        #video-stream {
            border: 1px solid #ddd;
            max-width: 100%;
            height: auto;
        }
        .stats-container {
            margin-top: 20px;
            padding: 15px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .control-buttons {
            margin-top: 20px;
            text-align: center;
        }
        #status-message {
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">24小时视频运动分析系统</h1>
        
        <div class="row">
            <div class="col-md-8">
                <div class="video-container">
                    <img id="video-stream" src="/video_feed" alt="视频流">
                </div>
                
                <div class="control-buttons">
                    <button id="start-btn" class="btn btn-success me-2">开始</button>
                    <button id="stop-btn" class="btn btn-danger">停止</button>
                </div>
                
                <div id="status-message"></div>
            </div>
            
            <div class="col-md-4">
                <div class="stats-container">
                    <h3>实时统计</h3>
                    <div id="stats-panel">
                        <p><strong>帧率:</strong> <span id="fps">0.0</span> FPS</p>
                        <p><strong>帧计数:</strong> <span id="frame-count">0</span></p>
                        <p><strong>特征总数:</strong> <span id="features-count">0</span></p>
                        <div id="feature-types"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // 定期更新统计信息
        function updateStats() {
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('frame-count').textContent = data.frame_count;
                    document.getElementById('features-count').textContent = data.features_count;
                    
                    // 更新特征类型
                    let featureTypesHtml = '';
                    for (const [type, count] of Object.entries(data.feature_types)) {
                        featureTypesHtml += `<p><strong>${type}:</strong> ${count}</p>`;
                    }
                    document.getElementById('feature-types').innerHTML = featureTypesHtml;
                })
                .catch(error => console.error('获取统计信息出错:', error));
        }
        
        // 每秒更新一次统计信息
        setInterval(updateStats, 1000);
        
        // 显示状态消息
        function showStatus(message, isSuccess) {
            const statusEl = document.getElementById('status-message');
            statusEl.textContent = message;
            statusEl.style.display = 'block';
            statusEl.className = isSuccess ? 'alert alert-success' : 'alert alert-danger';
            
            // 3秒后自动隐藏
            setTimeout(() => {
                statusEl.style.display = 'none';
            }, 3000);
        }
        
        // 控制按钮事件
        document.getElementById('start-btn').addEventListener('click', function() {
            this.disabled = true; // 防止重复点击
            document.getElementById('stop-btn').disabled = false;
            
            fetch('/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                showStatus('处理已启动', true);
                this.disabled = false;
            })
            .catch(error => {
                console.error('错误:', error);
                showStatus('启动处理失败：' + error, false);
                this.disabled = false;
            });
        });
        
        document.getElementById('stop-btn').addEventListener('click', function() {
            this.disabled = true; // 防止重复点击
            document.getElementById('start-btn').disabled = false;
            
            fetch('/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                showStatus('处理已停止', true);
                this.disabled = false;
            })
            .catch(error => {
                console.error('错误:', error);
                showStatus('停止处理失败：' + error, false);
                this.disabled = false;
            });
        });
        
        // 页面加载完成后，自动开始处理
        window.onload = function() {
            document.getElementById('start-btn').click();
        };
    </script>
</body>
</html>
        ''')
    
    # 启动视频处理线程
    processing_thread = threading.Thread(target=video_processing_thread, args=(args,))
    processing_thread.daemon = True
    processing_thread.start()
    
    # 启动Flask服务器
    logger.info(f"启动Web服务器，地址: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main() 