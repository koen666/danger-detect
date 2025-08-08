#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复配置文件编码问题的工具
此脚本会检查配置文件的编码，并确保它们正确保存为UTF-8格式
"""

import os
import sys
import json
import codecs
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def detect_encoding(file_path):
    """尝试检测文件编码"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'iso-8859-1', 'latin1']
    
    for encoding in encodings:
        try:
            with codecs.open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                return encoding, content
        except UnicodeDecodeError:
            continue
    
    return None, None

def fix_file_encoding(file_path, target_encoding='utf-8'):
    """修复文件编码为UTF-8"""
    logger.info(f"正在检查文件: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return False
    
    # 检测原始编码
    detected_encoding, content = detect_encoding(file_path)
    
    if not detected_encoding:
        logger.error(f"无法检测文件编码: {file_path}")
        return False
    
    logger.info(f"检测到文件编码: {detected_encoding}")
    
    if detected_encoding.lower() == target_encoding.lower():
        logger.info(f"文件已经是 {target_encoding} 编码，无需修复")
        return True
    
    # 尝试解析JSON格式，确保文件内容有效
    try:
        json_data = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"无效的JSON格式: {str(e)}")
        return False
    
    # 保存为UTF-8编码
    try:
        with codecs.open(file_path, 'w', encoding=target_encoding) as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logger.info(f"文件已重新保存为 {target_encoding} 编码")
        return True
    except Exception as e:
        logger.error(f"保存文件时出错: {str(e)}")
        return False

def main():
    """主函数"""
    config_files = [
        'config/rules.json',
        'config/notification.json'
    ]
    
    success_count = 0
    total_count = len(config_files)
    
    for file_path in config_files:
        if fix_file_encoding(file_path):
            success_count += 1
    
    logger.info(f"处理完成: {success_count}/{total_count} 个文件已修复")
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main()) 