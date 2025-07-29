import logging
import os

os.makedirs('./log',exist_ok=True)

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('./log/test.log'),logging.StreamHandler()])

logger = logging.getLogger(__name__)

logger.debug("debug test")
logger.info("info test")
logger.warning("warning test")
logger.error("error test")