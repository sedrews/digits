from itertools import chain
import time

from .main import *


class SamplingEvaluator(Evaluator):

    # Default numbers of samples are chosen such that Hoeffding bounds give:
    # 73778 samples -> 95% confidence of within .005
    # 10000 samples -> 95% confidence of within .0136
    def __init__(self, sampler, post, orig_prog, num=(10000,73778)):
        self.post = post
        self.sampler = sampler
        self.orig_prog = orig_prog
        self.fast_num = num[0]
        self.num = num[1]

        class Stats:
            def __init__(self):
                self.post_calls = {True : 0, False : 0}
                self.post_time = {True : 0, False : 0}
                self.error_calls = {True : 0, False : 0}
                self.error_time = {True : 0, False : 0}
            def as_array(self):
                return [("fast_post_calls", self.post_calls[True]), \
                        ("fast_post_time", self.post_time[True]), \
                        ("fast_error_calls", self.error_calls[True]), \
                        ("fast_error_time", self.error_time[True]), \
                        ("slow_post_calls", self.post_calls[False]), \
                        ("slow_post_time", self.post_time[False]), \
                        ("slow_error_calls", self.error_calls[False]), \
                        ("slow_error_time", self.error_time[False])]
            def __str__(self):
                return reduce(lambda x,y: x + y, [p[0]+":"+str(p[1]) for p in self.as_array()])
        self.stats = Stats()

    def get_stats(self):
        return [str(i) for i in chain(*self.stats.as_array())]

    def compute_post(self, prog, fast=True):
        start_time = time.time()
        self.stats.post_calls[fast] += 1
        samples = [self.sampler.get(j) for j in range(self.fast_num if fast else self.num)]
        trials = [prog.event_call(*sample) for sample in samples] # prog is parse.EventFunc
        event_map = {}
        def Pr(event):
            t = tuple(sorted(event.items()))
            if t not in event_map:
                counter = 0
                for trial in trials:
                    flag = True
                    for (e,b) in t:
                        if trial[e] != b:
                            flag = False
                            break
                    if flag:
                        counter += 1
                event_map[t] = counter / len(samples)
            return event_map[t]
        try:
            res = self.post(Pr)
        except ZeroDivisionError: # Since post might contain conditional probabilities
            res = False # For now -- in the future could use tristate
        self.stats.post_time[fast] += time.time() - start_time
        return res

    def compute_error(self, prog, fast=True):
        start_time = time.time()
        self.stats.error_calls[fast] += 1
        samples = [self.sampler.get(j) for j in range(self.fast_num if fast else self.num)]
        counter = 0
        for sample in samples:
            counter += 1 if prog(*sample) != self.orig_prog(*sample) else 0
        self.stats.error_time[fast] += time.time() - start_time
        return counter / len(samples)
