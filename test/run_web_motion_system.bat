@echo off
echo 启动Web版视频运动分析系统...
python motion_web_system.py ^
    --source 0 ^
    --width 320 ^
    --height 240 ^
    --use_optical_flow ^
    --optical_flow_method farneback ^
    --host 127.0.0.1 ^
    --port 5000
pause 