from itertools import chain
import time

from .main import *
from .tracking import Stats


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

        self._stats = Stats(fast_post_calls = 0, fast_post_time = 0,
                            slow_post_calls = 0, slow_post_time = 0,
                            fast_error_calls = 0, fast_error_time = 0,
                            slow_error_calls = 0, slow_error_time = 0)

    @property
    def stats(self):
        return self._stats

    def compute_post(self, prog, fast=True):
        start_time = time.time()
        res = self._compute_post(prog, fast)
        total_time = time.time() - start_time
        if fast:
            self.stats.fast_post_calls += 1
            self.stats.fast_post_time += total_time
        else:
            self.stats.slow_post_calls += 1
            self.stats.slow_post_time += total_time
        return res

    def _compute_post(self, prog, fast):
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
        return res

    def compute_error(self, prog, fast=True):
        start_time = time.time()
        res = self._compute_error(prog, fast)
        total_time = time.time() - start_time
        if fast:
            self.stats.fast_error_calls += 1
            self.stats.fast_error_time += total_time
        else:
            self.stats.slow_error_calls += 1
            self.stats.slow_error_time += total_time
        return res

    def _compute_error(self, prog, fast=True):
        samples = [self.sampler.get(j) for j in range(self.fast_num if fast else self.num)]
        counter = 0
        for sample in samples:
            counter += 1 if prog(*sample) != self.orig_prog(*sample) else 0
        return counter / len(samples)
