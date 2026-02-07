import logging
from pythonjsonlogger.json import JsonFormatter
from datetime import datetime

def create_logger(name = __name__):
    logger = logging.Logger(name)
    logger.setLevel(logging.INFO)
    json_formatter = JsonFormatter(json_indent=2, json_ensure_ascii=False)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    file_handler.setFormatter(json_formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger