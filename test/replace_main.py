#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Replace main.py with a clean UTF-8 version
"""

import os
import sys
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def replace_main_file():
    """Replace the original main.py with our clean version"""
    try:
        # Check if our clean version exists
        if not os.path.exists("main_clean.py"):
            logger.error("main_clean.py does not exist. Cannot proceed with replacement.")
            return False
            
        # Create backup of original main.py if it exists
        if os.path.exists("main.py"):
            backup_file = "main.py.bak"
            i = 1
            while os.path.exists(backup_file):
                backup_file = f"main.py.bak.{i}"
                i += 1
                
            logger.info(f"Creating backup of original main.py as {backup_file}")
            shutil.copy2("main.py", backup_file)
            
        # Replace main.py with our clean version
        logger.info("Replacing main.py with the clean UTF-8 version")
        shutil.copy2("main_clean.py", "main.py")
        
        logger.info("Replacement completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during replacement: {str(e)}")
        return False

if __name__ == "__main__":
    if replace_main_file():
        logger.info("Successfully replaced main.py with clean version")
        sys.exit(0)
    else:
        logger.error("Failed to replace main.py")
        sys.exit(1) 