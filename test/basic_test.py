#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic test script to verify encoding and logging are working
"""

import os
import sys
import logging
import json
import codecs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_config_files():
    """Test the encoding of configuration files"""
    logger.info("Testing configuration file encoding...")
    
    # Test rules.json
    try:
        config_path = "config/rules.json"
        if os.path.exists(config_path):
            logger.info(f"Reading {config_path}...")
            with codecs.open(config_path, 'r', 'utf-8') as f:
                rules = json.load(f)
                logger.info(f"Successfully read {config_path}. Found {len(rules)} rules.")
        else:
            logger.warning(f"{config_path} does not exist.")
    except Exception as e:
        logger.error(f"Error reading {config_path}: {str(e)}")
    
    # Test notification.json
    try:
        config_path = "config/notification.json"
        if os.path.exists(config_path):
            logger.info(f"Reading {config_path}...")
            with codecs.open(config_path, 'r', 'utf-8') as f:
                config = json.load(f)
                logger.info(f"Successfully read {config_path}.")
        else:
            logger.warning(f"{config_path} does not exist.")
    except Exception as e:
        logger.error(f"Error reading {config_path}: {str(e)}")
    
    logger.info("Config file test completed.")

def test_module_imports():
    """Test importing key modules"""
    logger.info("Testing key module imports...")
    
    try:
        from models.video.video_capture import VideoCaptureManager
        logger.info("Successfully imported VideoCaptureManager")
    except Exception as e:
        logger.error(f"Error importing VideoCaptureManager: {str(e)}")
    
    try:
        from models.motion.motion_manager import MotionFeatureManager
        logger.info("Successfully imported MotionFeatureManager")
    except Exception as e:
        logger.error(f"Error importing MotionFeatureManager: {str(e)}")
    
    try:
        from models.trajectory.trajectory_manager import TrajectoryManager
        logger.info("Successfully imported TrajectoryManager")
    except Exception as e:
        logger.error(f"Error importing TrajectoryManager: {str(e)}")
    
    try:
        from models.alert.rule_analyzer import RuleAnalyzer
        logger.info("Successfully imported RuleAnalyzer")
    except Exception as e:
        logger.error(f"Error importing RuleAnalyzer: {str(e)}")
    
    logger.info("Module import test completed.")

def main():
    """Main entry point"""
    logger.info("Starting basic test script...")
    
    test_config_files()
    test_module_imports()
    
    logger.info("Basic test completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 