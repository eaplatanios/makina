import inspect
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(sys.stdout)],
                    format='%(asctime)-15s - %(levelname)-7s - %(name)-20s - %(message)s')
logger = logging.getLogger(inspect.currentframe().f_back.f_globals['__name__'])
