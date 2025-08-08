#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复main.py文件编码问题的工具
"""

import sys
import codecs
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def fix_main_py_encoding():
    """修复main.py文件的编码问题"""
    try:
        # 尝试以不同编码读取文件
        for encoding in ['gbk', 'gb2312', 'gb18030', 'utf-8-sig', 'latin1', 'iso-8859-1']:
            try:
                logger.info(f"尝试以 {encoding} 编码读取main.py...")
                with codecs.open('main.py', 'r', encoding=encoding) as f:
                    content = f.read()
                
                # 替换可能出现问题的字符
                content = content.replace('主程�?', '主程序')
                content = content.replace('告警功能�?', '告警功能。')
                content = content.replace('处�?', '处理')
                content = content.replace('系�?', '系统')
                content = content.replace('管�?', '管道')
                content = content.replace('关系图模�?', '关系图模型')
                content = content.replace('线程和�?', '线程和锁')
                content = content.replace('状态字�?', '状态字典')
                content = content.replace('告警系�?', '告警系统')
                
                # 以UTF-8格式保存文件
                logger.info("将修复后的文件保存为UTF-8编码...")
                with codecs.open('main.py', 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info("成功修复main.py文件编码！")
                return True
            except UnicodeDecodeError:
                logger.warning(f"{encoding} 编码解码失败，尝试下一个编码...")
                continue
        
        logger.error("无法以任何支持的编码读取main.py文件")
        return False
    except Exception as e:
        logger.error(f"修复main.py文件编码时出错: {str(e)}")
        return False

if __name__ == "__main__":
    success = fix_main_py_encoding()
    sys.exit(0 if success else 1) 