@echo off
echo 启动危险动作检测验证脚本...
python test_danger_detection.py ^
    --source 0 ^
    --width 640 ^
    --height 480 ^
    --record ^
    --save_alerts
pause 