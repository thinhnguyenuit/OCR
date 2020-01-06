from function import count, count1, sc
import os
import logging
from time import time
from functools import partial
from multiprocessing.pool import Pool


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('requests').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

def main():
    ts = time()
    # sc(0)
    # sc(1)
    # sc(2)
    # sc(3)
    # sc(4)
    # sc(5)
    # sc(6)
    # sc(7)
    # sc(8)
    # sc(9)
    # sc(10)
    # sc(11)
    # sc(12)
    # sc(13)
    r = range(0,15)
    with Pool(4) as p:
        p.map(sc, r)
    logging.info('time: %s', time() - ts)

if __name__ == '__main__':
    main()