@echo off
echo 启动AI增强版危险行为监控系统...

REM 设置API密钥
set OPENAI_API_KEY=your_openai_api_key_here

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
    --save_alerts ^
    --enable_ai ^
    --vision_model yolov8n ^
    --ai_interval 20 ^
    --scene_analysis

pause 