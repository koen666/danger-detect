#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频监控系统 - 配置管理模块
用于管理系统配置和参数
"""

import os
import json
import yaml

class Config:
    """配置管理类"""
    
    @staticmethod
    def load_config(config_path):
        """加载配置文件
        
        Args:
            config_path: 配置文件路径，支持json和yaml格式
            
        Returns:
            配置字典
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        _, ext = os.path.splitext(config_path)
        
        if ext.lower() in ('.yaml', '.yml'):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif ext.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")
    
    @staticmethod
    def save_config(config, config_path):
        """保存配置到文件
        
        Args:
            config: 配置字典
            config_path: 配置文件保存路径
        """
        _, ext = os.path.splitext(config_path)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        if ext.lower() in ('.yaml', '.yml'):
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif ext.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")

__all__ = ['Config'] 