from digits import *


class SamplingEvaluator(Evaluator):

    # post 's prob predicates need to take (Sample, output) tuples as input
    def __init__(self, sampler, post, orig_prog, num_samples=2000):
        self.post = post
        self.sampler = sampler
        self.orig_prog = orig_prog
        self.num_samples = num_samples

    def compute_post(self, prog):
        samples = [self.sampler.next_sample() for j in range(self.num_samples)]
        trials = [(sample, prog(sample)) for sample in samples]
        counter = {event : 0 for event in self.post.events}
        for trial in trials:
            res = self.post.evaluate_events(trial)
            for event,value in res.items():
                counter[event] += 1 if value else 0
        estimates = {event : counter[event] / self.num_samples for event in counter}
        return self.post.check(estimates)

    def compute_error(self, prog):
        samples = [self.sampler.next_sample() for j in range(self.num_samples)]
        counter = 0
        for sample in samples:
            counter += 1 if prog(sample) != self.orig_prog(sample) else 0
        return counter / self.num_samples
