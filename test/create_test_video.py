#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
创建测试视频文件
生成一个简单的动画，包含移动的物体，用于测试运动检测算法
"""

import os
import sys
import logging
import cv2
import numpy as np
import time
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='创建测试视频')
    parser.add_argument('--output', type=str, default='test_video.mp4', help='输出视频文件名')
    parser.add_argument('--width', type=int, default=640, help='视频宽度')
    parser.add_argument('--height', type=int, default=480, help='视频高度')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    parser.add_argument('--duration', type=int, default=10, help='视频时长（秒）')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置视频编码器和输出对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
    
    if not out.isOpened():
        logger.error(f"无法创建视频文件: {args.output}")
        return
    
    # 计算总帧数
    total_frames = args.duration * args.fps
    
    logger.info(f"开始创建测试视频: {args.width}x{args.height}@{args.fps}fps, {args.duration}秒")
    
    # 创建多个移动的物体
    objects = [
        {
            'type': 'circle',
            'radius': 30,
            'color': (255, 0, 0),  # 蓝色
            'speed': (3, 2),
            'position': [100, 100]
        },
        {
            'type': 'rectangle',
            'size': (60, 40),
            'color': (0, 255, 0),  # 绿色
            'speed': (-2, 3),
            'position': [300, 200]
        },
        {
            'type': 'circle',
            'radius': 20,
            'color': (0, 0, 255),  # 红色
            'speed': (4, -3),
            'position': [500, 300]
        },
        {
            'type': 'rectangle',
            'size': (80, 50),
            'color': (255, 255, 0),  # 黄色
            'speed': (-3, -2),
            'position': [400, 400]
        }
    ]
    
    start_time = time.time()
    
    # 逐帧生成视频
    for i in range(total_frames):
        # 创建空白帧
        frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        
        # 绘制背景 - 简单的网格
        for x in range(0, args.width, 50):
            cv2.line(frame, (x, 0), (x, args.height), (30, 30, 30), 1)
        for y in range(0, args.height, 50):
            cv2.line(frame, (0, y), (args.width, y), (30, 30, 30), 1)
        
        # 绘制并更新每个物体的位置
        for obj in objects:
            # 更新位置
            obj['position'][0] += obj['speed'][0]
            obj['position'][1] += obj['speed'][1]
            
            # 边界检查和反弹
            if obj['type'] == 'circle':
                r = obj['radius']
                if obj['position'][0] - r < 0 or obj['position'][0] + r > args.width:
                    obj['speed'] = (-obj['speed'][0], obj['speed'][1])
                if obj['position'][1] - r < 0 or obj['position'][1] + r > args.height:
                    obj['speed'] = (obj['speed'][0], -obj['speed'][1])
            else:  # rectangle
                w, h = obj['size']
                if obj['position'][0] < 0 or obj['position'][0] + w > args.width:
                    obj['speed'] = (-obj['speed'][0], obj['speed'][1])
                if obj['position'][1] < 0 or obj['position'][1] + h > args.height:
                    obj['speed'] = (obj['speed'][0], -obj['speed'][1])
            
            # 绘制物体
            if obj['type'] == 'circle':
                cv2.circle(frame, (int(obj['position'][0]), int(obj['position'][1])), 
                          obj['radius'], obj['color'], -1)
            else:  # rectangle
                x, y = int(obj['position'][0]), int(obj['position'][1])
                w, h = obj['size']
                cv2.rectangle(frame, (x, y), (x + w, y + h), obj['color'], -1)
        
        # 添加一些随机运动 - 每5帧创建一个新的随机圆点
        if i % 5 == 0:
            x = np.random.randint(0, args.width)
            y = np.random.randint(0, args.height)
            radius = np.random.randint(5, 15)
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.circle(frame, (x, y), radius, color, -1)
        
        # 添加帧计数器
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 写入帧
        out.write(frame)
        
        # 显示进度
        if i % (total_frames // 10) == 0:
            progress = i / total_frames * 100
            logger.info(f"进度: {progress:.1f}% - 已处理 {i}/{total_frames} 帧")
    
    # 释放资源
    out.release()
    
    elapsed = time.time() - start_time
    logger.info(f"测试视频创建完成: {args.output}")
    logger.info(f"总耗时: {elapsed:.2f}秒 ({total_frames / elapsed:.2f} FPS)")

if __name__ == "__main__":
    main() 