#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版视频分析系统 - 用于测试
"""

import os
import cv2
import time
import argparse
import logging

# 视频捕获和处理
from models.video.video_capture import VideoCaptureManager
from models.video.frame_processor import FrameProcessor

# 运动特征提取
from models.motion.motion_manager import MotionFeatureManager

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="简化版视频分析系统")
    
    # 视频源
    parser.add_argument("--source", type=str, default="0", help="视频源 (0表示摄像头)")
    
    # 视频参数
    parser.add_argument("--width", type=int, default=640, help="帧宽度")
    parser.add_argument("--height", type=int, default=480, help="帧高度")
    parser.add_argument("--fps", type=int, default=30, help="目标FPS")
    
    # 处理选项
    parser.add_argument("--denoise", action="store_true", help="启用降噪")
    parser.add_argument("--enhance", action="store_true", help="启用图像增强")
    parser.add_argument("--use_optical_flow", action="store_true", help="使用光流特征提取")
    parser.add_argument("--use_motion_history", action="store_true", help="使用运动历史特征提取")
    
    # 其他选项
    parser.add_argument("--use_gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化视频捕获管理器
    capture_manager = VideoCaptureManager(
        source=args.source,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    
    # 连接视频源
    if not capture_manager.connect():
        logger.error("无法连接到视频源")
        return
    
    # 初始化帧处理器 - 使用正确的构造函数参数
    frame_processor = FrameProcessor(
        preprocessing_config={
            'resize_dim': (args.width, args.height),
            'normalize': False,  # 保持uint8格式用于显示
            'denoise': args.denoise,
            'denoise_method': 'gaussian',
            'enhance': args.enhance,
            'equalize_hist': False
        }
    )
    
    # 初始化运动特征管理器
    motion_manager = MotionFeatureManager(
        use_optical_flow=args.use_optical_flow,
        use_motion_history=args.use_motion_history,
        use_gpu=args.use_gpu
    )
    
    # 创建窗口
    cv2.namedWindow("Video Analysis", cv2.WINDOW_NORMAL)
    
    # 处理视频帧
    frame_count = 0
    start_time = time.time()
    previous_frame = None
    
    logger.info("开始处理视频帧...")
    
    try:
        while True:
            # 读取帧
            ret, frame = capture_manager.read()
            if not ret or frame is None:
                logger.warning("读取帧失败")
                break
            
            # 处理帧
            processed_frame = frame_processor.process_frame(frame)
            
            # 提取运动特征
            motion_features = motion_manager.extract_features(processed_frame, previous_frame)
            
            # 更新上一帧
            previous_frame = processed_frame.copy()
            
            # 可视化运动特征
            if motion_features:
                processed_frame = motion_manager.visualize_features(processed_frame, motion_features)
            
            # 计算并显示FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # 在帧上绘制FPS
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示帧
            cv2.imshow("Video Analysis", processed_frame)
            
            # 按ESC键退出
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('s'):  # 按s键保存当前帧
                filename = os.path.join(args.output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(filename, processed_frame)
                logger.info(f"已保存帧: {filename}")
                
    except KeyboardInterrupt:
        logger.info("用户中断")
    
    finally:
        # 释放资源
        capture_manager.release()
        cv2.destroyAllWindows()
        
        # 打印统计信息
        logger.info(f"总共处理了 {frame_count} 帧")
        logger.info(f"平均FPS: {frame_count / (time.time() - start_time):.2f}")

if __name__ == "__main__":
    main() 