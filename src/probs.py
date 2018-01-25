from scipy.stats import norm
from random import random


def gaussian(mean, variance):
    return norm.rvs(mean, variance**0.5)

def step(bars):
    # bars is a list of (min,max,pmass)
    assert sum([bar[2] for bar in bars]) == 1.
    r = random()
    total = 0.
    for bar in bars:
        total += bar[2]
        if r <= total:
            return random() * (bar[1] - bar[0]) + bar[0]
    assert False, "This statement is reached with 0 probability"

prob_dict = {'gaussian' : gaussian, 'step' : step}
