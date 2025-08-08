@echo off
echo 启动危险行为监控系统...

python integrate_danger_detection.py ^
    --source 0 ^
    --width 640 ^
    --height 480 ^
    --process_every 3 ^
    --max_fps 30 ^
    --feature_threshold 80 ^
    --area_threshold 0.05 ^
    --alert_cooldown 10 ^
    --min_confidence 0.5 ^
    --alert_region "[(50,50), (590,50), (590,430), (50,430)]" ^
    --record ^
    --save_alerts

pause 