from digits import *


class SamplingEvaluator(Evaluator):

    def __init__(self, sampler, post, num_samples=2000):
        self.post = post
        self.sampler = sampler
        self.num_samples = num_samples

    def compute_post(self, prog):
        samples = [self.sampler.next_sample() for j in range(self.num_samples)]
        samples = [(*sample, prog(*sample)) for sample in samples] # Include the output value
        samples = [dict(zip(range(len(sample)), sample)) for sample in samples] # XXX requires the same indexing for the variable values as in the definitions of the events in post
        counter = {event : 0 for event in self.post.probs}
        for sample in samples:
            for event,value in post.evaluate_prob_events(sample).items():
                counter[event] += 1 if value else 0
        estimates = {event : counter[event] / len(samples) for event in counter}
        return post.evaluate_expression(estimates)

    def compute_error(self, prog):
        pass
