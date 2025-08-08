@echo off
echo 启动视频运动分析系统...
python motion_video_system.py ^
    --source output/test_motion.mp4 ^
    --width 640 ^
    --height 480 ^
    --use_optical_flow ^
    --use_motion_history ^
    --optical_flow_method farneback ^
    --save_frames ^
    --save_interval 10 ^
    --display
pause 