import inspect
import logging
import sys
from contextlib import contextmanager
from timeit import default_timer

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(sys.stdout)],
                    format='%(asctime)-15s - %(levelname)-7s - %(name)-20s - %(message)s')
logger = logging.getLogger(inspect.currentframe().f_back.f_globals['__name__'])


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapsed = lambda: default_timer() - start
    yield lambda: elapsed()
    end = default_timer()
    elapsed = lambda: end - start
