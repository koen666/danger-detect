#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
告警处理器模块
负责根据规则分析处理告警事件。
"""

import os
import logging
import json
from datetime import datetime, timedelta
import threading
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class Alert:
    """告警类，表示一个告警事件"""
    
    def __init__(self, rule_name, description, severity="低", frame=None, metadata=None):
        """初始化告警事件"""
        self.rule_name = rule_name
        self.description = description
        self.severity = severity
        self.timestamp = datetime.now()
        self.frame = frame
        self.metadata = metadata or {}
        self.id = f"{rule_name}_{self.timestamp.strftime('%Y%m%d%H%M%S%f')}"
    
    def to_dict(self):
        """将告警转换为字典"""
        return {
            "id": self.id,
            "rule_name": self.rule_name,
            "description": self.description,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def __str__(self):
        """字符串表示"""
        return f"Alert({self.rule_name}, {self.severity}, {self.timestamp})"
    
    def save_frame(self, output_dir="alerts"):
        """保存告警帧"""
        if self.frame is None:
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/alert_{self.id}.jpg"
        
        try:
            cv2.imwrite(filename, self.frame)
            logger.debug(f"保存告警帧至 {filename}")
            return filename
        except Exception as e:
            logger.error(f"保存告警帧失败: {str(e)}")
            return None

class AlertProcessor:
    """
    告警处理器类，负责处理和管理告警事件。
    包括告警生成、存储、排重和通知触发。
    """
    
    def __init__(self, min_alert_interval=10.0, max_alerts=1000, output_dir="alerts"):
        """初始化告警处理器"""
        self.min_alert_interval = min_alert_interval
        self.max_alerts = max_alerts
        self.output_dir = output_dir
        self.alerts = []
        self.alert_counts = {}  # 按规则名称计数
        self.last_alert_time = {}  # 按规则名称记录最后告警时间
        self.lock = threading.RLock()
        self.alert_handlers = []
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def process(self, alert_data):
        """处理告警数据"""
        if isinstance(alert_data, dict):
            # 如果是字典，则创建告警对象
            rule_name = alert_data.get("rule_name", "未知规则")
            description = alert_data.get("description", "")
            severity = alert_data.get("severity", "低")
            frame = alert_data.get("frame")
            metadata = alert_data.get("metadata", {})
            
            alert = Alert(rule_name, description, severity, frame, metadata)
        elif isinstance(alert_data, Alert):
            # 如果已经是告警对象，则直接使用
            alert = alert_data
        else:
            logger.error(f"不支持的告警数据类型: {type(alert_data)}")
            return None
        
        # 检查是否应该处理该告警
        if not self._should_process(alert):
            return None
        
        # 更新计数和时间戳
        self._update_alert_stats(alert)
        
        # 保存告警到列表
        with self.lock:
            self.alerts.append(alert)
            # 限制最大告警数量
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
        
        # 保存告警帧
        if alert.frame is not None:
            alert.save_frame(self.output_dir)
        
        # 记录告警
        logger.info(f"处理告警: {alert}")
        
        # 调用所有已注册的处理器
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"调用告警处理器失败: {str(e)}")
        
        return alert
    
    def _should_process(self, alert):
        """检查是否应该处理该告警"""
        rule_name = alert.rule_name
        current_time = datetime.now()
        
        with self.lock:
            # 检查最小间隔
            if rule_name in self.last_alert_time:
                last_time = self.last_alert_time[rule_name]
                if (current_time - last_time).total_seconds() < self.min_alert_interval:
                    logger.debug(f"告警 '{rule_name}' 间隔太短，跳过")
                    return False
        
        return True
    
    def _update_alert_stats(self, alert):
        """更新告警统计信息"""
        rule_name = alert.rule_name
        current_time = datetime.now()
        
        with self.lock:
            # 更新最后告警时间
            self.last_alert_time[rule_name] = current_time
            
            # 更新计数
            if rule_name not in self.alert_counts:
                self.alert_counts[rule_name] = 0
            self.alert_counts[rule_name] += 1
    
    def register_handler(self, handler):
        """注册告警处理器"""
        if callable(handler) and handler not in self.alert_handlers:
            self.alert_handlers.append(handler)
            return True
        return False
    
    def unregister_handler(self, handler):
        """取消注册告警处理器"""
        if handler in self.alert_handlers:
            self.alert_handlers.remove(handler)
            return True
        return False
    
    def get_alerts(self, start_time=None, end_time=None, rule_name=None, severity=None, limit=100):
        """获取告警列表"""
        filtered_alerts = []
        
        # 如果没有指定起始时间，则使用过去24小时
        if start_time is None:
            start_time = datetime.now() - timedelta(days=1)
        
        # 如果没有指定结束时间，则使用当前时间
        if end_time is None:
            end_time = datetime.now()
        
        with self.lock:
            for alert in reversed(self.alerts):  # 从最新到最旧排序
                # 应用过滤条件
                if alert.timestamp < start_time or alert.timestamp > end_time:
                    continue
                
                if rule_name is not None and alert.rule_name != rule_name:
                    continue
                
                if severity is not None and alert.severity != severity:
                    continue
                
                filtered_alerts.append(alert)
                
                # 限制返回数量
                if len(filtered_alerts) >= limit:
                    break
        
        return filtered_alerts
    
    def get_alert_counts(self, start_time=None, end_time=None, group_by="rule"):
        """获取告警计数统计"""
        # 如果没有指定起始时间，则使用过去24小时
        if start_time is None:
            start_time = datetime.now() - timedelta(days=1)
        
        # 如果没有指定结束时间，则使用当前时间
        if end_time is None:
            end_time = datetime.now()
        
        counts = {}
        with self.lock:
            for alert in self.alerts:
                if alert.timestamp < start_time or alert.timestamp > end_time:
                    continue
                
                # 根据分组类型统计
                if group_by == "rule":
                    key = alert.rule_name
                elif group_by == "severity":
                    key = alert.severity
                elif group_by == "hour":
                    key = alert.timestamp.strftime("%Y-%m-%d %H:00")
                elif group_by == "day":
                    key = alert.timestamp.strftime("%Y-%m-%d")
                else:
                    key = "total"
                
                if key not in counts:
                    counts[key] = 0
                counts[key] += 1
        
        return counts
    
    def clear_alerts(self, older_than=None):
        """清除告警"""
        if older_than is None:
            # 默认清除所有告警
            with self.lock:
                self.alerts = []
            logger.info("清除所有告警")
            return True
        
        # 清除特定时间之前的告警
        threshold_time = datetime.now() - older_than
        with self.lock:
            self.alerts = [alert for alert in self.alerts if alert.timestamp >= threshold_time]
        
        logger.info(f"清除 {threshold_time} 之前的告警")
        return True
    
    def save_alerts(self, filename="alerts_export.json"):
        """保存告警到文件"""
        path = os.path.join(self.output_dir, filename)
        with self.lock:
            alerts_data = [alert.to_dict() for alert in self.alerts]
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(alerts_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存 {len(alerts_data)} 条告警至 {path}")
            return True
        except Exception as e:
            logger.error(f"保存告警失败: {str(e)}")
            return False
    
    def load_alerts(self, filename="alerts_export.json"):
        """从文件加载告警"""
        path = os.path.join(self.output_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"告警文件不存在: {path}")
            return False
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                alerts_data = json.load(f)
            
            loaded_alerts = []
            for data in alerts_data:
                alert = Alert(
                    rule_name=data["rule_name"],
                    description=data["description"],
                    severity=data["severity"],
                    metadata=data.get("metadata", {})
                )
                alert.id = data.get("id", alert.id)
                alert.timestamp = datetime.fromisoformat(data["timestamp"])
                loaded_alerts.append(alert)
            
            with self.lock:
                self.alerts = loaded_alerts
            
            logger.info(f"加载 {len(loaded_alerts)} 条告警从 {path}")
            return True
        except Exception as e:
            logger.error(f"加载告警失败: {str(e)}")
            return False 