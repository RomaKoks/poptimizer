import subprocess
import sys
import traceback
from datetime import datetime

from poptimizer.__main__ import evolve, optimize
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
    time_thresh = 6
    if datetime.today().hour > time_thresh:
        opt()

    first = True
    while first or (time_thresh - 4 <= datetime.today().hour < time_thresh) or (datetime.today().hour == 23):
        print('NOW is', datetime.today())
        first = False
        try:
            evolve()
        except Exception as e:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            del exc_info
            print(e)
            if 'unspecified launch failure' in str(e):
                break

    if time_thresh <= datetime.today().hour < time_thresh + 2:
        opt()
        shut_down = ["shutdown", "-f", "-s", "-t", "30"]
        subprocess.call(shut_down)
