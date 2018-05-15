from scipy.stats import norm
from random import random
from fractions import Fraction


def gaussian(mean, variance):
    return norm.rvs(mean, variance**0.5)

def step(bars):
    # bars is a list of (min,max,pmass)
    sanity = sum([bar[2] for bar in bars])
    assert 1. - sanity < 0.00001 , "bars sum to " + str(sanity)
    r = random()
    total = Fraction(0)
    for bar in bars:
        total += Fraction(bar[2])
        if r <= total:
            return random() * (bar[1] - bar[0]) + bar[0]
    assert False, "This statement is reached with 0 probability"

prob_dict = {'gaussian' : gaussian, 'step' : step}
