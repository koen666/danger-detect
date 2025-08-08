#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
极简化的测试脚本，只测试最基本的MotionHistoryExtractor功能
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
        logging.FileHandler("motion_test_simple.log")
    ]
)
logger = logging.getLogger(__name__)

def test_motion_history():
    """测试运动历史特征提取"""
    logger.info("开始测试MotionHistoryExtractor...")
    
    try:
        # 导入模块
        from models.motion.motion_history import MotionHistoryExtractor
        logger.info("导入MotionHistoryExtractor成功")
        
        # 初始化
        extractor = MotionHistoryExtractor(history_length=20, threshold=30)
        logger.info("初始化MotionHistoryExtractor成功")
        logger.info(f"is_initialized: {extractor.is_initialized}")
        
        # 创建测试帧
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 绘制移动对象
        cv2.rectangle(frame1, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(frame2, (150, 150), (250, 250), (255, 255, 255), -1)
        
        logger.info("测试帧创建成功")
        
        # 处理第一帧
        logger.info("处理第一帧")
        features1 = extractor.extract(frame1, None)
        logger.info(f"第一帧特征数: {len(features1)}")
        
        # 处理第二帧
        logger.info("处理第二帧")
        features2 = extractor.extract(frame2, frame1)
        logger.info(f"第二帧特征数: {len(features2)}")
        
        # 获取运动历史
        mhi = extractor.get_motion_history_image()
        logger.info(f"运动历史图像类型: {type(mhi)}, 形状: {mhi.shape if mhi is not None else 'None'}")
        
        # 显示和保存结果
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        if mhi is not None:
            # 保存运动历史图像
            vis_mhi = np.clip(mhi * 10, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, "simple_mhi.jpg"), vis_mhi)
            logger.info("保存了运动历史图像")
        
        logger.info("测试完成")
        return True
    
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_motion_history()
    sys.exit(0 if success else 1) 