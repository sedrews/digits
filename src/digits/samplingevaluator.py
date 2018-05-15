from digits import *


class SamplingEvaluator(Evaluator):

    def __init__(self, sampler, post, orig_prog, num_samples=2000):
        self.post = post
        self.sampler = sampler
        self.orig_prog = orig_prog
        self.num_samples = num_samples

    def compute_post(self, prog):
        samples = [self.sampler.get(j) for j in range(self.num_samples)]
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
                event_map[t] = counter / self.num_samples
            return event_map[t]
        try:
            res = self.post(Pr)
        except ZeroDivisionError: # Since post might contain conditional probabilities
            res = False # For now -- in the future could use tristate
        return res

    def compute_error(self, prog):
        samples = [self.sampler.get(j) for j in range(self.num_samples)]
        counter = 0
        for sample in samples:
            counter += 1 if prog(*sample) != self.orig_prog(*sample) else 0
        return counter / self.num_samples
