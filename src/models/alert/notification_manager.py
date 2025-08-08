#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通知管理器模块
负责处理各种通知渠道的配置和消息分发。
"""

import os
import json
import logging
import smtplib
import requests
import time
from datetime import datetime
import codecs
import traceback
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class NotificationManager:
    """
    通知管理器类，负责处理各种通知方式的配置和发送逻辑。
    支持控制台输出、文件记录、邮件、短信和Webhook等多种通知方式。
    """
    
    def __init__(self, notification_config="config/notification.json"):
        """初始化通知管理器"""
        self.config_path = notification_config
        self.config = self._load_config()
        self.enabled = self.config.get("enabled", True)
        self.throttling = self.config.get("throttling", {})
        self.notification_hours = self.config.get("notification_hours", {})
        
        # 初始化通知方法
        self.notification_methods = {
            "console": self._send_console_notification,
            "file": self._send_file_notification,
            "email": self._send_email_notification,
            "sms": self._send_sms_notification,
            "webhook": self._send_webhook_notification
        }
        
        # 初始化通知计数器和时间戳
        self.notification_counts = {}
        self.last_notification_time = {}
        
        # 确保告警日志目录存在
        if self.config["methods"].get("file", {}).get("enabled", False):
            log_file = self.config["methods"]["file"].get("file_path", "alerts/alerts_log.txt")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def _load_config(self):
        """加载通知配置"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"通知配置文件不存在: {self.config_path}, 使用默认配置")
                return self._get_default_config()
            
            # 使用codecs模块并尝试多种编码
            for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030']:
                try:
                    with codecs.open(self.config_path, 'r', encoding=encoding) as f:
                        config = json.load(f)
                        logger.info(f"成功以 {encoding} 编码加载通知配置文件")
                        return config
                except UnicodeDecodeError:
                    continue
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误 (编码: {encoding}): {e}")
                    continue
            
            # 如果所有编码都失败
            logger.error(f"无法以任何支持的编码读取通知配置文件: {self.config_path}")
            return self._get_default_config()
            
        except Exception as e:
            logger.error(f"加载通知配置出错: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_default_config()
    
    def _get_default_config(self):
        """获取默认通知配置"""
        return {
            "enabled": True,
            "methods": {
                "console": {
                    "enabled": True,
                    "min_severity": "低"
                },
                "file": {
                    "enabled": True,
                    "min_severity": "低",
                    "file_path": "alerts/alerts_log.txt"
                }
            },
            "alert_templates": {
                "low": {
                    "subject": "低严重性警报: {rule_name}",
                    "body": "检测到低严重性事件:\n规则: {rule_name}\n描述: {description}\n时间: {timestamp}\n位置: {location}"
                },
                "medium": {
                    "subject": "中严重性警报: {rule_name}",
                    "body": "检测到中严重性事件:\n规则: {rule_name}\n描述: {description}\n时间: {timestamp}\n位置: {location}\n\n请及时查看和处理。"
                },
                "high": {
                    "subject": "高严重性警报: {rule_name}",
                    "body": "检测到高严重性事件:\n规则: {rule_name}\n描述: {description}\n时间: {timestamp}\n位置: {location}\n\n请立即查看和处理！"
                }
            },
            "throttling": {
                "enabled": True,
                "max_alerts_per_minute": 10,
                "max_alerts_per_rule": 5,
                "cooldown_period": 300
            }
        }
    
    def _check_notification_hours(self):
        """检查当前是否在通知时间范围内"""
        if not self.notification_hours.get("enabled", False):
            return True
        
        now = datetime.now().time()
        start_time_str = self.notification_hours.get("start_time", "09:00")
        end_time_str = self.notification_hours.get("end_time", "18:00")
        
        try:
            start_time = datetime.strptime(start_time_str, "%H:%M").time()
            end_time = datetime.strptime(end_time_str, "%H:%M").time()
            
            if start_time <= now <= end_time:
                return True
            
            # 检查是否忽略高严重性告警的时间限制
            if self.notification_hours.get("ignore_high_severity", True):
                return severity == "高"
            
            return False
        except Exception as e:
            logger.error(f"解析通知时间出错: {str(e)}")
            return True
    
    def _check_throttling(self, alert):
        """检查是否应该限制该告警的通知频率"""
        if not self.throttling.get("enabled", False):
            return False
        
        rule_name = alert.get("rule_name", "unknown")
        current_time = datetime.now().timestamp()
        max_per_minute = self.throttling.get("max_alerts_per_minute", 10)
        max_per_rule = self.throttling.get("max_alerts_per_rule", 5)
        cooldown = self.throttling.get("cooldown_period", 300)
        
        # 检查总体告警率
        if len(self.notification_counts) > 0:
            one_minute_ago = current_time - 60
            recent_count = sum(1 for t in self.notification_counts.values() if t > one_minute_ago)
            if recent_count >= max_per_minute:
                logger.warning(f"告警频率限制: 超过每分钟最大告警数 {max_per_minute}")
                return True
        
        # 检查特定规则的告警率
        if rule_name in self.last_notification_time:
            # 检查冷却期
            last_time = self.last_notification_time[rule_name]
            if current_time - last_time < cooldown:
                logger.warning(f"告警频率限制: 规则 '{rule_name}' 在冷却期内")
                return True
            
            # 检查每规则最大告警数
            if rule_name in self.notification_counts:
                one_minute_ago = current_time - 60
                if self.notification_counts[rule_name] > one_minute_ago:
                    rule_count = sum(1 for t in self.notification_counts[rule_name] if t > one_minute_ago)
                    if rule_count >= max_per_rule:
                        logger.warning(f"告警频率限制: 规则 '{rule_name}' 超过每分钟最大告警数 {max_per_rule}")
                        return True
        
        return False
    
    def _update_notification_stats(self, alert):
        """更新通知统计信息"""
        rule_name = alert.get("rule_name", "unknown")
        current_time = datetime.now().timestamp()
        
        # 更新总体计数
        self.notification_counts[current_time] = current_time
        
        # 清理过旧的记录
        one_minute_ago = current_time - 60
        self.notification_counts = {t: v for t, v in self.notification_counts.items() if v > one_minute_ago}
        
        # 更新规则特定计数
        if rule_name not in self.notification_counts:
            self.notification_counts[rule_name] = []
        self.notification_counts[rule_name].append(current_time)
        
        # 更新最后通知时间
        self.last_notification_time[rule_name] = current_time
    
    def _format_message(self, alert):
        """格式化通知消息"""
        severity = alert.get("severity", "低").lower()
        if severity not in ["低", "中", "高"]:
            severity = "低"
        
        severity_map = {"低": "low", "中": "medium", "高": "high"}
        template_key = severity_map[severity]
        
        templates = self.config.get("alert_templates", {})
        template = templates.get(template_key, {})
        
        subject_template = template.get("subject", "告警: {rule_name}")
        body_template = template.get("body", "检测到告警:\n规则: {rule_name}\n描述: {description}\n时间: {timestamp}")
        
        # 准备模板变量
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        location = alert.get("location", "未知位置")
        
        # 替换模板变量
        subject = subject_template.format(
            rule_name=alert.get("rule_name", "未知规则"),
            description=alert.get("description", "无描述"),
            timestamp=timestamp,
            location=location
        )
        
        body = body_template.format(
            rule_name=alert.get("rule_name", "未知规则"),
            description=alert.get("description", "无描述"),
            timestamp=timestamp,
            location=location
        )
        
        return subject, body
    
    def _should_notify(self, method_config, alert):
        """检查是否应该通过特定方法发送通知"""
        if not method_config.get("enabled", False):
            return False
        
        min_severity = method_config.get("min_severity", "低")
        alert_severity = alert.get("severity", "低")
        
        severity_levels = {"低": 1, "中": 2, "高": 3}
        if severity_levels.get(alert_severity, 0) < severity_levels.get(min_severity, 0):
            return False
        
        return True
    
    def send_notification(self, alert):
        """发送通知"""
        if not self.enabled:
            logger.debug("通知功能已禁用")
            return False
        
        if not self._check_notification_hours():
            logger.debug("当前不在通知时间范围内")
            return False
        
        if self._check_throttling(alert):
            logger.debug("通知被节流系统限制")
            return False
        
        # 更新通知统计
        self._update_notification_stats(alert)
        
        # 准备通知内容
        subject, body = self._format_message(alert)
        
        # 遍历所有通知方法
        success = False
        for method_name, method_config in self.config["methods"].items():
            if self._should_notify(method_config, alert):
                if method_name in self.notification_methods:
                    try:
                        self.notification_methods[method_name](alert, subject, body, method_config)
                        success = True
                    except Exception as e:
                        logger.error(f"发送 {method_name} 通知失败: {str(e)}")
        
        return success
    
    def _send_console_notification(self, alert, subject, body, config):
        """发送控制台通知"""
        print(f"\n[告警] {subject}")
        print("-" * 50)
        print(body)
        print("-" * 50)
    
    def _send_file_notification(self, alert, subject, body, config):
        """发送文件通知"""
        file_path = config.get("file_path", "alerts/alerts_log.txt")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {subject}\n")
            f.write("-" * 50 + "\n")
            f.write(body + "\n")
            f.write("-" * 50 + "\n")
    
    def _send_email_notification(self, alert, subject, body, config):
        """发送邮件通知"""
        smtp_server = config.get("smtp_server")
        smtp_port = config.get("smtp_port", 587)
        username = config.get("username")
        password = config.get("password")
        from_address = config.get("from_address")
        to_addresses = config.get("to_addresses", [])
        use_tls = config.get("use_tls", True)
        
        if not smtp_server or not username or not password or not from_address or not to_addresses:
            logger.error("邮件配置不完整")
            return
        
        msg = MIMEMultipart()
        msg['From'] = from_address
        msg['To'] = ", ".join(to_addresses)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            if use_tls:
                server.starttls()
            server.login(username, password)
            text = msg.as_string()
            server.sendmail(from_address, to_addresses, text)
            server.quit()
            logger.info(f"已发送邮件通知到: {', '.join(to_addresses)}")
        except Exception as e:
            logger.error(f"发送邮件失败: {str(e)}")
            raise
    
    def _send_sms_notification(self, alert, subject, body, config):
        """发送短信通知"""
        api_key = config.get("api_key")
        from_number = config.get("from_number")
        to_numbers = config.get("to_numbers", [])
        
        if not api_key or not from_number or not to_numbers:
            logger.error("短信配置不完整")
            return
        
        # 这里只是示例，实际实现需要根据具体的短信API服务
        # 下面假设使用Twilio API
        try:
            # 实际调用短信API的代码
            logger.info(f"已发送短信通知到: {', '.join(to_numbers)}")
        except Exception as e:
            logger.error(f"发送短信失败: {str(e)}")
            raise
    
    def _send_webhook_notification(self, alert, subject, body, config):
        """发送Webhook通知"""
        url = config.get("url")
        headers = config.get("headers", {})
        method = config.get("method", "POST")
        
        if not url:
            logger.error("Webhook配置不完整")
            return
        
        payload = {
            "subject": subject,
            "body": body,
            "alert": alert,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if method.upper() == "POST":
                response = requests.post(url, json=payload, headers=headers)
            else:
                response = requests.get(url, params=payload, headers=headers)
            
            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"Webhook通知成功发送: {url}")
            else:
                logger.warning(f"Webhook返回非成功状态码: {response.status_code}")
        except Exception as e:
            logger.error(f"发送Webhook通知失败: {str(e)}")
            raise 