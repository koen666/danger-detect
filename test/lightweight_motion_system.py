#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
轻量级视频运动分析系统 - 专注于流畅性和低延迟
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
from flask import Flask, Response, render_template, request, jsonify

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("lightweight_motion.log")
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
frame_buffer = None
processed_buffer = None
features_count = 0
feature_types = {}
fps = 0
frame_count = 0
running = True
processing_active = True
lock = threading.Lock()
capture_thread = None
processing_thread = None

# 创建Flask应用
app = Flask(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='轻量级视频运动分析系统')
    
    # 视频源参数
    parser.add_argument('--source', type=str, default='0', help='视频源 (0表示摄像头, 或者是视频文件路径)')
    parser.add_argument('--width', type=int, default=320, help='视频宽度')
    parser.add_argument('--height', type=int, default=240, help='视频高度')
    
    # 运动特征参数
    parser.add_argument('--use_optical_flow', action='store_true', help='启用光流特征提取')
    parser.add_argument('--optical_flow_method', type=str, default='farneback', 
                       choices=['farneback', 'lucas_kanade'], help='光流算法')
    
    # Web服务器参数
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Web服务器主机')
    parser.add_argument('--port', type=int, default=5000, help='Web服务器端口')
    
    # 优化参数
    parser.add_argument('--process_every', type=int, default=3, help='每处理N帧中的1帧，用于提高流畅性')
    parser.add_argument('--max_fps', type=int, default=30, help='最大目标FPS')
    
    return parser.parse_args()

def capture_thread_func(args):
    """专门用于捕获视频帧的线程，与处理分离以提高流畅性"""
    global frame_buffer, running, frame_count
    
    try:
        # 打开视频源
        if args.source.isdigit():
            source = int(args.source)
        else:
            source = args.source
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            return
        
        # 设置视频参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲区
        
        # 可能的额外优化（根据系统支持情况）
        try:
            # 尝试设置为MJPG格式，这在某些相机上可以提高性能
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            # 尝试禁用自动对焦，减少处理延迟
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except:
            pass  # 忽略不支持的设置
            
        logger.info(f"成功打开视频源: {source}")
        
        # 捕获循环
        local_count = 0
        start_time = time.time()
        
        while running:
            # 捕获帧，丢弃旧帧
            for _ in range(2):  # 快速丢弃缓冲区中的旧帧
                ret, _ = cap.read()
            
            # 获取最新的帧
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and not source.isdigit():
                    # 如果是视频文件，循环播放
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.error("无法读取视频帧")
                    break
            
            # 更新帧缓冲区
            with lock:
                frame_buffer = frame.copy()
                local_count += 1
                frame_count = local_count
            
            # 计算和限制帧率
            elapsed = time.time() - start_time
            if elapsed > 0:
                capture_fps = local_count / elapsed
                if capture_fps > args.max_fps:
                    time.sleep(1.0 / args.max_fps - 1.0 / capture_fps)
        
        # 清理资源
        cap.release()
        logger.info("捕获线程已终止")
        
    except Exception as e:
        logger.error(f"捕获线程出错: {str(e)}")
        logger.error(traceback.format_exc())
        running = False

def process_thread_func(args):
    """专门用于处理视频帧的线程"""
    global frame_buffer, processed_buffer, features_count, feature_types, fps, processing_active
    
    try:
        # 导入必要模块
        try:
            from models.motion.motion_manager import MotionFeatureManager
            logger.info("成功导入MotionFeatureManager")
        except ImportError as e:
            logger.error(f"导入MotionFeatureManager失败: {str(e)}")
            return
        
        # 初始化运动特征管理器（简化配置）
        motion_manager = MotionFeatureManager(
            use_optical_flow=args.use_optical_flow,
            use_motion_history=False,  # 禁用运动历史以提高性能
            optical_flow_method=args.optical_flow_method,
            use_gpu=True  # 尝试使用GPU
        )
        
        # 初始化变量
        prev_frame = None
        local_frame_count = 0
        start_time = time.time()
        skip_counter = 0
        process_scale = 0.5  # 处理时将帧缩小到一半尺寸
        
        while running:
            # 如果已暂停处理，则休眠
            if not processing_active:
                time.sleep(0.1)
                continue
            
            current_frame = None
            with lock:
                if frame_buffer is not None:
                    current_frame = frame_buffer.copy()
            
            if current_frame is None:
                time.sleep(0.01)
                continue
            
            # 跳过部分帧来提高性能
            skip_counter += 1
            if skip_counter < args.process_every:
                # 仍然更新处理过的帧缓冲区，但不做特征提取
                with lock:
                    processed_buffer = current_frame.copy()
                continue
            
            skip_counter = 0
            
            # 降低分辨率进行处理
            small_frame = cv2.resize(current_frame, (0, 0), fx=process_scale, fy=process_scale)
            
            # 处理第一帧的情况
            if prev_frame is None:
                prev_frame = small_frame.copy()
                with lock:
                    processed_buffer = current_frame.copy()
                continue
            
            # 提取运动特征
            features = motion_manager.extract_features(small_frame, prev_frame)
            
            # 可视化特征
            vis_frame = current_frame.copy()
            if features:
                # 调整特征坐标回原始尺寸
                for feature in features:
                    if hasattr(feature, 'position'):
                        feature.position = (int(feature.position[0] / process_scale), 
                                          int(feature.position[1] / process_scale))
                    if hasattr(feature, 'magnitude'):
                        feature.magnitude = feature.magnitude / process_scale
                
                vis_frame = motion_manager.visualize_features(vis_frame, features)
            
            # 统计特征类型
            local_feature_types = {}
            for feature in features:
                feature_type = feature.type
                if feature_type in local_feature_types:
                    local_feature_types[feature_type] += 1
                else:
                    local_feature_types[feature_type] = 1
            
            # 显示简化的帧信息（减少文字渲染以提高性能）
            elapsed_time = time.time() - start_time
            local_fps = local_frame_count / elapsed_time if elapsed_time > 0 else 0
            
            cv2.putText(vis_frame, f"FPS: {local_fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(vis_frame, f"Features: {len(features)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 更新全局变量
            with lock:
                processed_buffer = vis_frame.copy()
                features_count = len(features)
                feature_types = local_feature_types.copy()
                fps = local_fps
            
            # 更新状态
            prev_frame = small_frame.copy()
            local_frame_count += 1
            
            # 短暂休眠以降低CPU使用率
            time.sleep(0.001)
        
        logger.info("处理线程已终止")
        
    except Exception as e:
        logger.error(f"处理线程出错: {str(e)}")
        logger.error(traceback.format_exc())
        running = False

def generate_frames():
    """生成视频帧流"""
    global processed_buffer, frame_buffer
    
    while running:
        # 优先使用处理过的帧，如果没有则使用原始帧
        current_frame = None
        with lock:
            if processed_buffer is not None:
                current_frame = processed_buffer.copy()
            elif frame_buffer is not None:
                current_frame = frame_buffer.copy()
        
        if current_frame is not None:
            # 压缩质量降低，提高传输效率
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode('.jpg', current_frame, encode_param)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 适当降低帧率，减轻浏览器负担
        time.sleep(0.01)

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
    global processing_active
    
    logger.info("收到开始处理请求")
    with lock:
        processing_active = True
    
    return jsonify({'status': 'success', 'message': '处理已启动'})

@app.route('/stop', methods=['POST'])
def stop_processing():
    """停止处理"""
    global processing_active
    
    logger.info("收到停止处理请求")
    with lock:
        processing_active = False
    
    return jsonify({'status': 'success', 'message': '处理已停止'})

def create_html_template():
    """创建HTML模板"""
    os.makedirs('templates', exist_ok=True)
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>轻量级视频运动分析系统</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 10px;
            background-color: #f8f9fa;
        }
        .video-container {
            margin-top: 10px;
            text-align: center;
        }
        #video-stream {
            border: 1px solid #ddd;
            max-width: 100%;
            height: auto;
        }
        .stats-container {
            margin-top: 10px;
            padding: 10px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.1);
        }
        .control-buttons {
            margin-top: 10px;
            text-align: center;
        }
        #status-message {
            margin-top: 5px;
            padding: 5px;
            border-radius: 3px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <h2 class="text-center">轻量级视频运动分析系统</h2>
        
        <div class="row">
            <div class="col-md-9">
                <div class="video-container">
                    <img id="video-stream" src="/video_feed" alt="视频流">
                </div>
                
                <div class="control-buttons">
                    <button id="start-btn" class="btn btn-success btn-sm me-2">开始处理</button>
                    <button id="stop-btn" class="btn btn-danger btn-sm">停止处理</button>
                </div>
                
                <div id="status-message"></div>
            </div>
            
            <div class="col-md-3">
                <div class="stats-container">
                    <h5>实时统计</h5>
                    <div id="stats-panel">
                        <p class="mb-1"><strong>帧率:</strong> <span id="fps">0.0</span> FPS</p>
                        <p class="mb-1"><strong>帧计数:</strong> <span id="frame-count">0</span></p>
                        <p class="mb-1"><strong>特征总数:</strong> <span id="features-count">0</span></p>
                        <div id="feature-types"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
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
                        featureTypesHtml += `<p class="mb-1"><strong>${type}:</strong> ${count}</p>`;
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
            statusEl.className = isSuccess ? 'alert alert-success py-1 px-2' : 'alert alert-danger py-1 px-2';
            
            // 2秒后自动隐藏
            setTimeout(() => {
                statusEl.style.display = 'none';
            }, 2000);
        }
        
        // 控制按钮事件
        document.getElementById('start-btn').addEventListener('click', function() {
            fetch('/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                showStatus('处理已启动', true);
            })
            .catch(error => {
                showStatus('操作失败', false);
            });
        });
        
        document.getElementById('stop-btn').addEventListener('click', function() {
            fetch('/stop', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'}
            })
            .then(response => response.json())
            .then(data => {
                showStatus('处理已停止', true);
            })
            .catch(error => {
                showStatus('操作失败', false);
            });
        });
    </script>
</body>
</html>
        ''')

def main():
    """主函数"""
    global capture_thread, processing_thread
    
    args = parse_args()
    
    # 创建HTML模板
    create_html_template()
    
    # 启动捕获线程
    capture_thread = threading.Thread(target=capture_thread_func, args=(args,))
    capture_thread.daemon = True
    capture_thread.start()
    
    # 启动处理线程
    processing_thread = threading.Thread(target=process_thread_func, args=(args,))
    processing_thread.daemon = True
    processing_thread.start()
    
    # 启动Flask服务器
    logger.info(f"启动Web服务器，地址: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main() 