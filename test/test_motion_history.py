#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化测试脚本，专门测试MotionHistoryExtractor功能
"""

import os
import sys
import logging
import cv2
import numpy as np
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("motion_history_test.log")
    ]
)
logger = logging.getLogger(__name__)

def test_motion_history():
    """测试运动历史特征提取"""
    logger.info("测试MotionHistoryExtractor...")
    
    try:
        from models.motion.motion_history import MotionHistoryExtractor
        
        # 初始化运动历史提取器
        motion_history = MotionHistoryExtractor(
            history_length=20,
            threshold=30
        )
        
        logger.info("MotionHistoryExtractor初始化成功.")
        logger.info(f"是否已初始化: {motion_history.is_initialized}")
        
        # 创建测试帧（简单的移动方块）
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 在第一帧上绘制方块
        cv2.rectangle(frame1, (100, 100), (150, 150), (255, 255, 255), -1)
        
        # 在第二帧上绘制移动后的方块
        cv2.rectangle(frame2, (120, 120), (170, 170), (255, 255, 255), -1)
        
        # 提取特征
        logger.info("提取运动历史特征...")
        features = motion_history.extract(frame2, frame1)
        
        logger.info(f"提取了 {len(features)} 个运动历史特征.")
        
        # 获取运动历史图像
        mhi = motion_history.get_motion_history_image()
        if mhi is not None:
            # 保存运动历史图像
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # 把浮点型运动历史图像转换为可视化格式
            vis_mhi = np.clip(mhi * 10, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, "motion_history.jpg"), vis_mhi)
            logger.info(f"保存了运动历史图像到output/motion_history.jpg")
        
        # 测试摄像头
        logger.info("尝试使用摄像头测试...")
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.warning("摄像头不可用，跳过摄像头测试")
                return
            
            logger.info("摄像头打开成功. 按 'q' 退出.")
            
            ret, prev_frame = cap.read()
            if not ret:
                logger.warning("无法从摄像头读取，跳过摄像头测试")
                cap.release()
                return
            
            for i in range(100):  # 最多处理100帧
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                # 提取特征
                features = motion_history.extract(curr_frame, prev_frame)
                
                # 获取运动历史图像用于显示
                mhi = motion_history.get_motion_history_image()
                if mhi is not None:
                    # 转换为可视化格式
                    vis_mhi = np.clip(mhi * 10, 0, 255).astype(np.uint8)
                    # 转换为彩色图像用于更好的可视化
                    vis_mhi_color = cv2.applyColorMap(vis_mhi, cv2.COLORMAP_JET)
                    
                    # 显示原始帧和运动历史
                    combined = np.hstack((curr_frame, vis_mhi_color))
                    cv2.imshow("Motion History", combined)
                    
                    # 每30帧保存一次
                    if i % 30 == 0:
                        output_path = os.path.join(output_dir, f"webcam_mhi_{i}.jpg")
                        cv2.imwrite(output_path, combined)
                        logger.info(f"保存帧到 {output_path}")
                
                # 检查是否按下q键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                prev_frame = curr_frame.copy()
            
            # 清理
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"摄像头测试错误: {str(e)}")
        
    except Exception as e:
        logger.error(f"测试运动历史特征提取错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """程序入口"""
    logger.info("开始运动历史测试脚本...")
    
    test_motion_history()
    
    logger.info("运动历史测试完成.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 