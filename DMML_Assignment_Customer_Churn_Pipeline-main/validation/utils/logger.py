import logging
import os
from datetime import datetime

class Logger:
    @staticmethod
    def get_logger(path="../logs", name:str=''):
        if not os.path.exists(path):
            os.makedirs(path)
        logger = logging.getLogger()
        # Create handlers
        console_handler = logging.StreamHandler()  # Logs to terminal
        filename =  f'{name}_log_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(os.path.join(path, filename))  # Logs to file
        # Set logging format
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        return logger

logger = Logger.get_logger(name="validation")
