#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试MotionFeatureManager模块
"""

import os
import sys
import logging
import cv2
import numpy as np
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("motion_manager_test.log")
    ]
)
logger = logging.getLogger(__name__)

def test_motion_manager():
    """测试MotionFeatureManager"""
    logger.info("开始测试MotionFeatureManager...")
    
    try:
        from models.motion.motion_manager import MotionFeatureManager
        logger.info("导入MotionFeatureManager成功")
        
        # 初始化，同时启用光流和运动历史
        motion_manager = MotionFeatureManager(
            use_optical_flow=True,
            use_motion_history=True,
            use_gpu=False,
            optical_flow_method='farneback',
            history_length=20,
            threshold=30
        )
        
        logger.info("MotionFeatureManager初始化成功")
        logger.info(f"is_initialized: {motion_manager.is_initialized}")
        logger.info(f"光流提取器已启用: {motion_manager.optical_flow_extractor is not None}")
        logger.info(f"运动历史提取器已启用: {motion_manager.motion_history_extractor is not None}")
        
        # 创建测试帧
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 绘制移动对象
        cv2.rectangle(frame1, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(frame2, (150, 150), (250, 250), (255, 255, 255), -1)
        
        logger.info("测试帧创建成功")
        
        # 处理第一帧
        logger.info("处理第一帧")
        features1 = motion_manager.extract_features(frame1, None)
        logger.info(f"第一帧特征数: {len(features1)}")
        
        # 处理第二帧
        logger.info("处理第二帧")
        features2 = motion_manager.extract_features(frame2, frame1)
        logger.info(f"第二帧特征数: {len(features2)}")
        
        # 分析特征类型
        feature_types = {}
        for feature in features2:
            feature_type = feature.type
            if feature_type in feature_types:
                feature_types[feature_type] += 1
            else:
                feature_types[feature_type] = 1
        
        logger.info("提取的特征类型统计:")
        for feature_type, count in feature_types.items():
            logger.info(f"  {feature_type}: {count}个特征")
        
        # 可视化特征
        vis_frame = motion_manager.visualize_features(frame2, features2)
        
        # 保存结果
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "motion_features.jpg"), vis_frame)
        logger.info("保存了特征可视化图像")
        
        # 测试摄像头
        logger.info("尝试使用摄像头测试...")
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.warning("摄像头不可用，跳过摄像头测试")
                return True
            
            logger.info("摄像头打开成功. 按 'q' 退出.")
            
            ret, prev_frame = cap.read()
            if not ret:
                logger.warning("无法从摄像头读取，跳过摄像头测试")
                cap.release()
                return True
            
            for i in range(100):  # 最多处理100帧
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                # 提取并可视化特征
                features = motion_manager.extract_features(curr_frame, prev_frame)
                vis_frame = motion_manager.visualize_features(curr_frame, features)
                
                # 显示结果
                cv2.imshow("Motion Features", vis_frame)
                
                # 每30帧保存一次
                if i % 30 == 0:
                    output_path = os.path.join(output_dir, f"webcam_motion_features_{i}.jpg")
                    cv2.imwrite(output_path, vis_frame)
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
        
        logger.info("测试完成")
        return True
    
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_motion_manager()
    sys.exit(0 if success else 1) 