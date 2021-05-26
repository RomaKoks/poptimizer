import sys
import traceback

from poptimizer.__main__ import evolve

if __name__ == '__main__':
    try:
        evolve()
    except Exception as e:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info
        print(e)
