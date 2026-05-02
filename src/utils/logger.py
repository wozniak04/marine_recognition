import logging
import os
from datetime import datetime
from pathlib import Path

def create_logger(name: str, log_dir: Path = Path("logs")) -> logging.Logger:    
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logging.basicConfig(
        level=logging.INFO, # only INFO, WARN, ERROR, CRIT
    )
    
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s]: [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    date = datetime.now().strftime("%Y-%m-%d")
    log_file_path = Path(log_dir, f"{date}.log")

    # Output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger