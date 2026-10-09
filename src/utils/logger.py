"""
Created By  : ...
Created Date: DD/MM/YYYY
Description : ...
"""
import logging


def get_logger(app_name="dbm"):
    """Create and configure logger.

    :return: logger
    :rtype: Logger
    """
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    return logger