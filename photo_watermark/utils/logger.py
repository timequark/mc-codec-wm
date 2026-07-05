"""统一日志。"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_logger(name="photo_watermark"):
    return logging.getLogger(name)
