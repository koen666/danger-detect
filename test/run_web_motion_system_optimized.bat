@echo off
echo 启动优化版Web视频运动分析系统...
python motion_web_system.py ^
    --source 0 ^
    --width 320 ^
    --height 240 ^
    --use_optical_flow ^
    --optical_flow_method farneback ^
    --save_frames ^
    --save_interval 30 ^
    --host 127.0.0.1 ^
    --port 5000
pause 