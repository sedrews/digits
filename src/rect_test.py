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
from probpost import ProbPost, Event

from z3 import *

from collections import namedtuple
import random

import sys
if len(sys.argv) > 1:
    random.seed(int(sys.argv[1]))
else:
    random.seed(0)

Sample = namedtuple('Sample', 'x y')
Holes = namedtuple('Holes', 'H1 H2 H3 H4')
def sketch(sample, holes):
    if holes.H1 < sample.x and sample.x < holes.H2 and holes.H3 < sample.y and sample.y < holes.H4:
        ret = 1
    else:
        ret = 0
    return ret
x,y = Reals('x y')
H1,H2,H3,H4 = Reals('H1 H2 H3 H4')
ret_0, ret_1, ret_2 = Reals('ret_0 ret_1 ret_2')
phi = And(Implies(And(H1 < x, x < H2, H3 < y, y < H4), \
                  And(ret_0 == 1, ret_2 == ret_0)), \
          Implies(Not(And(H1 < x, x < H2, H3 < y, y < H4)), \
                  And(ret_1 == 0, ret_2 == ret_1)))
phi = Exists([ret_0, ret_1], phi)

hole_defaults = Holes(-.5, .7, 0, 1)
orig_prog = lambda sample : sketch(sample, hole_defaults)

def precondition():
    rand = lambda : Fraction(random.random() * 2 - 1)
    return Sample(x=rand(), y=rand())

post = ProbPost()
post.preds = {"hired" : lambda iopair : iopair[1] == 1, \
              "minority" : lambda iopair : iopair[0].x < 0}
post.events = [Event({"hired" : True, "minority" : True}), \
               Event({"minority" : True}), \
               Event({"hired" : True, "minority" : False}), \
               Event({"minority" : False})]
def group_fair(pr_map):
    num = pr_map[Event({"hired" : True, "minority" : True})] / pr_map[Event({"minority" : True})]
    den = pr_map[Event({"hired" : True, "minority" : False})] / pr_map[Event({"minority" : False})]
    ratio = num / den
    return ratio > 0.95
post.func = group_fair

repair_model = SMTRepair(sketch, phi, ret_2, Holes)
evaluator = SamplingEvaluator(Sampler(precondition), post, orig_prog)

d = Digits(precondition, repair_model)
solns = d.repair(10, orig_prog, evaluator)
