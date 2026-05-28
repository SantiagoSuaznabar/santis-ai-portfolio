import logging
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def setup_logger():
    # Create a custom logger
    logger = logging.getLogger("rag_backend")
    

    log_level = os.getenv("LOG_LEVEL")
    logger.setLevel(getattr(logging, log_level, logging.DEBUG))

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    logging.getLogger("httpx").setLevel(logging.INFO)
    return logger

# Export the instantiated logger
logger = setup_logger()