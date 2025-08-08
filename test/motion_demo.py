#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运动特征检测演示程序
"""

import os
import sys
import logging
import cv2
import numpy as np
import time
import argparse
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("motion_demo.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运动特征检测演示程序')
    parser.add_argument('--source', type=str, default='0', help='视频源 (0表示摄像头, 或者是视频文件路径)')
    parser.add_argument('--width', type=int, default=640, help='视频宽度')
    parser.add_argument('--height', type=int, default=480, help='视频高度')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU加速（如果可用）')
    parser.add_argument('--optical_flow_method', type=str, default='farneback', choices=['farneback', 'lucas_kanade'], help='光流算法')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--save_interval', type=int, default=30, help='保存结果的帧间隔')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 导入运动特征管理器
    try:
        from models.motion.motion_manager import MotionFeatureManager
        logger.info("成功导入MotionFeatureManager")
    except ImportError as e:
        logger.error(f"导入MotionFeatureManager失败: {str(e)}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化运动特征管理器
    motion_manager = MotionFeatureManager(
        use_optical_flow=True,
        use_motion_history=True,
        optical_flow_method=args.optical_flow_method,
        use_gpu=args.use_gpu
    )
    
    # 尝试打开视频源
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
        ret, frame = cap.read()
        if not ret:
            logger.error("无法读取视频帧")
            return
        test_mode = False
    
    # 创建显示窗口
    cv2.namedWindow('Motion Features', cv2.WINDOW_NORMAL)
    
    # 初始化帧计数器和FPS计算
    frame_count = 0
    start_time = time.time()
    prev_frame = None
    
    logger.info("开始处理视频流")
    
    try:
        while True:
            # 获取帧
            if test_mode:
                # 在测试模式下生成移动的圆
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                center_x = int(args.width/2 + args.width/4 * np.sin(frame_count / 30.0))
                center_y = int(args.height/2 + args.height/4 * np.cos(frame_count / 30.0))
                cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            
            # 提取运动特征
            features = motion_manager.extract_features(frame, prev_frame)
            
            # 可视化特征
            vis_frame = frame.copy()
            
            try:
                vis_frame = motion_manager.visualize_features(frame, features)
            except Exception as e:
                logger.error(f"可视化特征时出错: {str(e)}")
            
            # 获取运动历史图像（如果可用）
            mhi = motion_manager.get_motion_history_image()
            if mhi is not None:
                try:
                    # 确保MHI是8位格式
                    if mhi.dtype != np.uint8:
                        # 归一化到0-255范围
                        mhi_norm = cv2.normalize(mhi, None, 0, 255, cv2.NORM_MINMAX)
                        mhi_8bit = np.uint8(mhi_norm)
                    else:
                        mhi_8bit = mhi
                    
                    # 转换为彩色显示
                    mhi_color = cv2.applyColorMap(mhi_8bit, cv2.COLORMAP_JET)
                    # 调整大小
                    mhi_color = cv2.resize(mhi_color, (frame.shape[1] // 3, frame.shape[0] // 3))
                    # 放在右上角
                    vis_frame[10:10+mhi_color.shape[0], vis_frame.shape[1]-mhi_color.shape[1]-10:vis_frame.shape[1]-10] = mhi_color
                except Exception as e:
                    logger.error(f"可视化运动历史图像时出错: {str(e)}")
            
            # 计算帧率
            curr_time = time.time()
            elapsed = curr_time - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # 显示帧率和特征数
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Features: {len(features)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示特征类型统计
            feature_types = {}
            for feature in features:
                feature_type = feature.type
                if feature_type in feature_types:
                    feature_types[feature_type] += 1
                else:
                    feature_types[feature_type] = 1
            
            y_pos = 90
            for feature_type, count in feature_types.items():
                cv2.putText(vis_frame, f"{feature_type}: {count}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
            
            # 显示结果
            cv2.imshow('Motion Features', vis_frame)
            
            # 保存结果
            if frame_count % args.save_interval == 0:
                output_path = os.path.join(args.output_dir, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(output_path, vis_frame)
                logger.info(f"保存帧到: {output_path}")
            
            # 更新
            prev_frame = frame.copy()
            frame_count += 1
            
            # 检查退出条件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q或Esc键退出
                break
    
    except Exception as e:
        logger.error(f"处理视频时出错: {str(e)}")
        logger.error(traceback.format_exc())
    
    finally:
        # 清理
        if not test_mode and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        
        # 显示统计信息
        elapsed = time.time() - start_time
        logger.info(f"处理完成: {frame_count} 帧在 {elapsed:.2f} 秒内 (平均 {frame_count/elapsed:.2f} FPS)")

if __name__ == "__main__":
    main() 