import json
import time


class Stats:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


start_time = None
jsonl_outfile = None

def start_timer():
    global start_time
    start_time = time.time()

def init_jsonl(filename):
    global jsonl_outfile
    jsonl_outfile = open(filename, 'w')

def log_event(event_name, **kwargs):
    global jsonl_outfile
    if jsonl_outfile is not None:
        json.dump({"timestamp" : time.time() - start_time, "event_name" : event_name, "params" : kwargs}, jsonl_outfile)
        jsonl_outfile.write('\n')
    else:
        print(time.time() - start_time, event_name, kwargs)

def log_stats(stats_name, stats_obj):
    if stats_obj is None:
        return
    assert isinstance(stats_obj, Stats)
    global jsonl_outfile
    if jsonl_outfile is not None:
        json.dump({"timestamp" : time.time() - start_time, "stats_name" : stats_name, "params" : stats_obj.__dict__}, jsonl_outfile)
        jsonl_outfile.write('\n')
    else:
        print(time.time() - start_time, stats_name, stats_obj.__dict__)
