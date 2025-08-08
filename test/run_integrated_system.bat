@echo off
echo 启动24小时视频监控告警系统...
python integrated_system.py ^
    --source output/test_motion.mp4 ^
    --width 640 ^
    --height 480 ^
    --use_optical_flow ^
    --use_motion_history ^
    --optical_flow_method farneback ^
    --save_frames ^
    --save_interval 15 ^
    --display
pause 