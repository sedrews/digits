import time


class Stats:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


start_time = None

def start_timer():
    global start_time
    start_time = time.time()

def log_event(event_name, **kwargs):
    print(time.time() - start_time, event_name, kwargs)

def log_stats(stats_name, stats_obj):
    if stats_obj is None:
        return
    assert isinstance(stats_obj, Stats)
    print(time.time() - start_time, stats_name, stats_obj.__dict__)
