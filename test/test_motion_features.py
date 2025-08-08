import argparse
import cv2
import numpy as np
from pathlib import Path
import time

from models.motion.motion_features import MotionFeatureManager
from models.video.video_capture import VideoCapture

def main():
    """
    运行测试应用程序，展示动态特征提取功能，包括�?
    - 光流分析
    - 运动历史
    """
    parser = argparse.ArgumentParser(description='测试运动特征提取')
    parser.add_argument('--source', type=str, default='0', help='视频�?(0 表示摄像�?')
    parser.add_argument('--use_optical_flow', action='store_true', help='使用光流分析')
    parser.add_argument('--use_motion_history', action='store_true', help='使用运动历史分析')
    parser.add_argument('--flow_method', type=str, default='farneback', choices=['farneback', 'pyr_lk'], 
                       help='光流计算方法 (farneback �?pyr_lk)')
    parser.add_argument('--output_dir', type=str, default='motion_samples', help='输出目录')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU加�?(如果可用)')
    parser.add_argument('--history_length', type=int, default=20, help='运动历史帧数')
    parser.add_argument('--threshold', type=int, default=30, help='运动检测阈�?)
    
    args = parser.parse_args()
    
    # 确保至少启用一种特征提取方�?
    if not args.use_optical_flow and not args.use_motion_history:
        print("至少需要启用一种特征提取方法，默认启用全部")
        args.use_optical_flow = True
        args.use_motion_history = True
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 初始化视频源
    cap = VideoCapture(args.source)
    
    # 初始化运动特征管理器
    motion_manager = MotionFeatureManager(
        use_optical_flow=args.use_optical_flow,
        use_motion_history=args.use_motion_history,
        optical_flow_method=args.flow_method,
        use_gpu=args.use_gpu,
        history_length=args.history_length,
        threshold=args.threshold
    )
    
    # 设置显示窗口
    cv2.namedWindow('Motion Features', cv2.WINDOW_NORMAL)
    
    # 处理视频�?
    frame_count = 0
    start_time = time.time()
    paused = False
    show_help = True
    
    while cap.is_opened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        
        if frame is None:
            continue
        
        # 提取运动特征
        features = motion_manager.extract_features(frame)
        
        # 可视化运动特�?
        vis_frame = motion_manager.visualize_features(frame, features)
        
        # 如果启用了运动历史，显示运动历史图像
        if args.use_motion_history:
            mhi = motion_manager.get_motion_history_image()
            if mhi is not None:
                # 转换为彩色图像以便于显示
                mhi_color = cv2.applyColorMap(mhi, cv2.COLORMAP_JET)
                # 调整大小以匹配原始帧
                mhi_color = cv2.resize(mhi_color, (frame.shape[1] // 3, frame.shape[0] // 3))
                # 放置在原始帧的右上角
                vis_frame[10:10+mhi_color.shape[0], vis_frame.shape[1]-mhi_color.shape[1]-10:vis_frame.shape[1]-10] = mhi_color
        
        # 显示帧率
        fps = frame_count / (time.time() - start_time + 1e-6)
        cv2.putText(vis_frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示特征计数
        cv2.putText(vis_frame, f'Features: {len(features)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示帮助信息
        if show_help:
            help_text = [
                "按键控制:",
                "Q/ESC - 退�?,
                "H - 显示/隐藏帮助",
                "空格 - 暂停/继续",
                "S - 保存当前�?,
                "R - 重置特征提取�?
            ]
            for i, text in enumerate(help_text):
                cv2.putText(vis_frame, text, (10, 110 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 显示结果
        cv2.imshow('Motion Features', vis_frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q �?ESC
            break
        elif key == ord(' '):  # 空格
            paused = not paused
        elif key == ord('h'):  # h
            show_help = not show_help
        elif key == ord('s'):  # s
            # 保存特征可视化图�?
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            save_path = output_dir / f'motion_features_{timestamp}.jpg'
            cv2.imwrite(str(save_path), vis_frame)
            print(f"已保存图像到: {save_path}")
            
            # 如果有运动历史图像，保存�?
            if args.use_motion_history:
                mhi = motion_manager.get_motion_history_image()
                if mhi is not None:
                    mhi_path = output_dir / f'motion_history_{timestamp}.jpg'
                    cv2.imwrite(str(mhi_path), mhi)
                    print(f"已保存运动历史到: {mhi_path}")
        elif key == ord('r'):  # r
            # 重置特征提取�?
            motion_manager.reset()
            frame_count = 0
            start_time = time.time()
            print("已重置特征提取器")
        
        frame_count += 1
    
    # 清理
    cap.release()
    cv2.destroyAllWindows()
    print(f"处理�?{frame_count} �?)

if __name__ == "__main__":
    main() 
