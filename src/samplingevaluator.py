from digits import *


class SamplingEvaluator(Evaluator):

    def __init__(self, sampler, post, num_samples=2000):
        self.post = post
        self.sampler = sampler
        self.num_samples = num_samples

    def compute_post(self, prog):
        # need a way to evaluate postcondition expressions in general
        pass

    def compute_error(self, prog):
        pass
