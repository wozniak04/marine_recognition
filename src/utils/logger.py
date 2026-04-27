import logging

def create_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logging.basicConfig(
        level=logging.INFO, # only INFO, WARN, ERROR, CRIT
        format="[%(asctime)s %(levelname)s]: [%(name)s] %(message)s"
    )
    
    return logger