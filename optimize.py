import sys
import traceback
from datetime import datetime

from poptimizer.__main__ import optimize
from poptimizer.data.views.listing import last_history_date


def opt():
    date = last_history_date()
    try:
        optimize(date)
    except Exception as e:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        del exc_info


if __name__ == '__main__':
    time_thresh = 8
    if datetime.today().hour > time_thresh:
        opt()