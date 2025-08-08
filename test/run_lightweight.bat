@echo off
echo 启动超轻量级视频运动分析系统...
python lightweight_motion_system.py ^
    --source 0 ^
    --width 320 ^
    --height 240 ^
    --use_optical_flow ^
    --optical_flow_method farneback ^
    --process_every 3 ^
    --max_fps 25 ^
    --host 127.0.0.1 ^
    --port 5000
pause 