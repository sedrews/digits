# Test usage of the DIGITS implementation
#
# Program model is inclusion in axis-aligned rectangles (in 2D)
# The input distribution is uniform over the square [-1,1] x [-1,1]
# The postcondition is a notion of group fairness that wants positive
#     x values to be as likely to be in the rectangle as non-positive
#     (i.e. the rectangles should have y=0 intersecting close to their center)

from digits import Digits, Sampler
from smtrepair import SMTRepair
from samplingevaluator import SamplingEvaluator
from fairexp import group_fairness

from z3 import *

from collections import namedtuple
from random import random

Sample = namedtuple('Sample', 'x y')
Holes = namedtuple('Holes', 'x_min x_max y_min y_max')
def sketch(sample, holes):
    if holes.x_min < sample.x and sample.x < holes.x_max:
        if holes.y_min < sample.y and sample.y < holes.y_max:
            return 1
    return 0
x,y = Reals('x y')
h1,h2,h3,h4 = Reals('x_min x_max y_min y_max')
out = Real('digitsoutput')
template = And( (out == 1) ==     And(h1 < x, x < h2, h3 < y, y < h4), \
                (out == 0) == Not(And(h1 < x, x < h2, h3 < y, y < h4)) )

orig_holes = Holes(-.5,.7,0,1)
orig_prog = lambda sample : sketch(sample, orig_holes)

def precondition():
    rand = lambda : random() * 2 - 1
    return Sample(x=rand(), y=rand())

postcondition = group_fairness(lambda iopair : iopair[1], lambda iopair : iopair[0].x > 0, 0.95)

repair_model = SMTRepair(sketch, template, out, Holes)
evaluator = SamplingEvaluator(Sampler(precondition), postcondition, orig_prog)

d = Digits(precondition, repair_model)
d.repair(5, orig_prog, evaluator)
