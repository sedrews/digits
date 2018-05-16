from digits import *


class SamplingEvaluator(Evaluator):

    # Default numbers of samples are chosen such that Hoeffding bounds give:
    # 73778 samples -> 95% confidence of within .005
    # 10000 samples -> 95% confidence of within .0136
    def __init__(self, sampler, post, orig_prog, fast_num=10000, num=73778):
        self.post = post
        self.sampler = sampler
        self.orig_prog = orig_prog
        self.fast_num = fast_num
        self.num = num

    def compute_post(self, prog, fast=True):
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
        samples = [self.sampler.get(j) for j in range(self.fast_num if fast else self.num)]
        counter = 0
        for sample in samples:
            counter += 1 if prog(*sample) != self.orig_prog(*sample) else 0
        return counter / len(samples)
