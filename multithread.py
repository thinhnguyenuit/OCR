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
    r = range(0,15)
    b = 0
    # for a in r:
    #    b = b + sc(r)
    # print(b)
    with Pool(4) as p:
       b = p.map(sc, r)
    print(b)
    logging.info('time: %s', time() - ts)

if __name__ == '__main__':
    main()