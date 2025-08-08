@echo off
echo 正在检查24小时视频监控告警系统模块...

echo 1. 检查可用模块
python connect_modules.py --check_modules
echo.

echo 2. 测试模块连接
python connect_modules.py --test_connection
echo.

echo 3. 列出系统依赖
python connect_modules.py --list_dependencies
echo.

echo 4. 测试运动特征模块
python connect_modules.py --test_motion
echo.

pause 