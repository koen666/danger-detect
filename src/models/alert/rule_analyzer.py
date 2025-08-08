#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
规则分析器模块
负责分析各种数据源并根据规则生成告警。
"""

import os
import json
import logging
from datetime import datetime
import codecs
import traceback

logger = logging.getLogger(__name__)

class RuleAnalyzer:
    """
    规则分析器类，负责加载和分析告警规则，并根据输入数据生成告警。
    支持基于行为、交互和场景的告警规则。
    """
    
    def __init__(self, rules_config="config/rules.json"):
        """初始化规则分析器"""
        self.rules_config_path = rules_config
        self.rules = self._load_rules()
        
    def _load_rules(self):
        """从配置文件加载规则"""
        try:
            if not os.path.exists(self.rules_config_path):
                logger.warning(f"规则配置文件不存在: {self.rules_config_path}, 使用默认规则")
                return []
            
            # 使用codecs模块并尝试多种编码
            for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030']:
                try:
                    with codecs.open(self.rules_config_path, 'r', encoding=encoding) as f:
                        config = json.load(f)
                        logger.info(f"成功以 {encoding} 编码加载规则配置文件")
                        rules_data = config.get("rules", [])
                        logger.info(f"加载了 {len(rules_data)} 条规则")
                        return rules_data
                except UnicodeDecodeError:
                    continue
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误 (编码: {encoding}): {e}")
                    continue
            
            # 如果所有编码都失败
            logger.error(f"无法以任何支持的编码读取规则配置文件: {self.rules_config_path}")
            return []
            
        except Exception as e:
            logger.error(f"加载规则配置出错: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def analyze(self, trajectories=None, behaviors=None, interactions=None, scene_graph=None, frame=None):
        """
        分析数据并根据规则生成告警
        
        Args:
            trajectories: 轨迹数据
            behaviors: 行为数据
            interactions: 交互数据
            scene_graph: 场景图数据
            frame: 当前帧
            
        Returns:
            生成的告警列表
        """
        alerts = []
        
        # 简化版实现 - 只处理行为告警
        if behaviors:
            for behavior in behaviors:
                # 查找匹配的规则
                for rule in self.rules:
                    if rule["type"] == "behavior":
                        conditions = rule.get("conditions", {})
                        
                        # 简单规则匹配
                        if hasattr(behavior, "behavior_type") and behavior.behavior_type.name == conditions.get("behavior_type"):
                            # 创建告警
                            alert = {
                                "rule_name": rule["name"],
                                "rule_id": rule["id"],
                                "description": rule["description"],
                                "severity": rule["severity"],
                                "timestamp": datetime.now().isoformat(),
                                "frame": frame
                            }
                            alerts.append(alert)
        
        # 简化版实现 - 处理交互告警
        if interactions:
            for interaction in interactions:
                # 查找匹配的规则
                for rule in self.rules:
                    if rule["type"] == "interaction":
                        conditions = rule.get("conditions", {})
                        
                        # 简单规则匹配
                        if hasattr(interaction, "behavior_type") and interaction.behavior_type.name == conditions.get("interaction_type"):
                            # 创建告警
                            alert = {
                                "rule_name": rule["name"],
                                "rule_id": rule["id"],
                                "description": rule["description"],
                                "severity": rule["severity"],
                                "timestamp": datetime.now().isoformat(),
                                "frame": frame
                            }
                            alerts.append(alert)
        
        # 简化版实现 - 处理场景告警
        if scene_graph and hasattr(scene_graph, "caption"):
            caption = scene_graph.caption
            
            # 查找匹配的规则
            for rule in self.rules:
                if rule["type"] == "semantic":
                    conditions = rule.get("conditions", {})
                    keywords = conditions.get("keywords", [])
                    
                    # 检查关键词匹配
                    matched_keywords = []
                    for keyword in keywords:
                        if keyword.lower() in caption.lower():
                            matched_keywords.append(keyword)
                    
                    # 如果有匹配的关键词，生成告警
                    if matched_keywords:
                        # 创建告警
                        alert = {
                            "rule_name": rule["name"],
                            "rule_id": rule["id"],
                            "description": rule["description"],
                            "severity": rule["severity"],
                            "timestamp": datetime.now().isoformat(),
                            "caption": caption,
                            "matched_keywords": matched_keywords,
                            "frame": frame
                        }
                        alerts.append(alert)
        
        return alerts 