#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
极简化的测试脚本，专门测试OpticalFlowExtractor功能
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
        logging.FileHandler("optical_flow_test.log")
    ]
)
logger = logging.getLogger(__name__)

def test_optical_flow():
    """测试光流特征提取"""
    logger.info("开始测试OpticalFlowExtractor...")
    
    try:
        # 导入模块
        from models.motion.optical_flow import OpticalFlowExtractor
        logger.info("导入OpticalFlowExtractor成功")
        
        # 初始化
        extractor = OpticalFlowExtractor(method='farneback', use_gpu=False)
        logger.info("初始化OpticalFlowExtractor成功")
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
        
        # 获取光流图像并可视化
        flow = extractor.get_flow()
        logger.info(f"光流类型: {type(flow)}, 形状: {flow.shape if flow is not None else 'None'}")
        
        # 显示和保存结果
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        if flow is not None:
            # 计算光流的幅值和方向
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # 创建HSV颜色空间的可视化
            hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = ang * 180 / np.pi / 2  # 色调根据方向
            hsv[..., 1] = 255  # 饱和度最大
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 亮度根据幅值
            
            # 转换为BGR颜色空间以显示
            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # 保存可视化结果
            cv2.imwrite(os.path.join(output_dir, "optical_flow.jpg"), flow_rgb)
            logger.info("保存了光流可视化图像")
            
            # 绘制光流向量
            vis_frame = frame2.copy()
            # 转为彩色图像以便可以绘制彩色向量
            if vis_frame.shape[2] == 1:
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)
                
            # 在每个网格点上绘制光流向量
            step = 16
            for y in range(step//2, flow.shape[0], step):
                for x in range(step//2, flow.shape[1], step):
                    fx, fy = flow[y, x]
                    # 只绘制有显著运动的点
                    if np.sqrt(fx*fx + fy*fy) > 1:
                        cv2.line(vis_frame, (x, y), (int(x+fx), int(y+fy)), (0, 255, 0), 1)
                        cv2.circle(vis_frame, (x, y), 1, (0, 255, 0), -1)
            
            # 保存向量可视化
            cv2.imwrite(os.path.join(output_dir, "optical_flow_vectors.jpg"), vis_frame)
            logger.info("保存了光流向量可视化图像")
        
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
                
                # 提取特征
                features = extractor.extract(curr_frame, prev_frame)
                
                # 获取光流用于显示
                flow = extractor.get_flow()
                if flow is not None:
                    # 计算光流的幅值和方向
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    
                    # 创建HSV颜色空间的可视化
                    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
                    hsv[..., 0] = ang * 180 / np.pi / 2  # 色调根据方向
                    hsv[..., 1] = 255  # 饱和度最大
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 亮度根据幅值
                    
                    # 转换为BGR颜色空间以显示
                    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    
                    # 在原始帧上绘制光流向量
                    vis_frame = curr_frame.copy()
                    step = 16
                    for y in range(step//2, flow.shape[0], step):
                        for x in range(step//2, flow.shape[1], step):
                            fx, fy = flow[y, x]
                            # 只绘制有显著运动的点
                            if np.sqrt(fx*fx + fy*fy) > 1:
                                cv2.line(vis_frame, (x, y), (int(x+fx), int(y+fy)), (0, 255, 0), 1)
                                cv2.circle(vis_frame, (x, y), 1, (0, 255, 0), -1)
                    
                    # 显示结果
                    combined = np.hstack((curr_frame, flow_rgb, vis_frame))
                    cv2.imshow("Optical Flow", combined)
                    
                    # 每30帧保存一次
                    if i % 30 == 0:
                        output_path = os.path.join(output_dir, f"webcam_flow_{i}.jpg")
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
        
        logger.info("测试完成")
        return True
    
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_optical_flow()
    sys.exit(0 if success else 1) 