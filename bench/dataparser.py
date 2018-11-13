from collections import namedtuple
import json


class RunStats:

    def __init__(self, jsonl_file):
        f = open(jsonl_file, 'r')
        self.data = [json.loads(line) for line in f.readlines()]
        f.close()

    @property
    def depth(self): # The largest depth completed
        res = -1
        for j in self.data:
            if j.get("event_name") == "finished depth":
                d = j["params"]["depth"]
                assert d > res
                res = d
        return res

    @property
    def best(self): # The best error found
        res = None
        for j in self.data:
            if j.get("event_name") == "new best":
                e = j["params"]["error"]
                assert res is None or e < res
                res = e
        return res

    @property
    def temporal_error_data(self): # List of (time, depth, error) tuples for each best error update
        T = namedtuple('TemporalError', ['time', 'depth', 'error'])
        tuples = []
        current_depth = 0
        for j in self.data:
            if j.get("event_name") == "finished depth":
                current_depth = j["params"]["depth"]
            elif j.get("event_name") == "new best":
                t = T(time=j["timestamp"], depth=current_depth, error=j["params"]["error"])
                tuples.append(t)
        return tuples

    @property
    def temporal_depth_data(self): # List of (time, depth) tuples for each finished depth
        T = namedtuple('TemporalDepth', ['time', 'depth'])
        tuples = []
        for j in self.data:
            if j.get("event_name") == "finished depth":
                tuples.append(T(time=j["timestamp"], depth=j["params"]["depth"]))
        return tuples


if __name__ == '__main__':
    s = RunStats("therm/res/therm_u10_b4.jsonl")
    print("depth:", s.depth, "best:", s.best)
    print(s.temporal_data)
