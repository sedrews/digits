# Test usage of the DIGITS implementation
#
# Program model is inclusion in axis-aligned rectangles (in 2D)
# The input distribution is uniform over the square [-1,1] x [-1,1]
# The postcondition is a notion of group fairness that wants positive
#     x values to be as likely to be in the rectangle as non-positive
#     (i.e. the rectangles should have y=0 intersecting close to their center)

from parse import parse_fr
from digits import Digits, Sampler
from smtrepair import SMTRepair
from samplingevaluator import SamplingEvaluator

from collections import namedtuple

prog_string='''
def pre():
    x = step([(-1, 1, 1)])
    y = step([(-1, 1, 1)])
    return x, y

def D(x, y):
    if Hole(-.5) < x and x < Hole(.7) and Hole(0) < y and y < Hole(1):
        ret = 1
    else:
        ret = 0
    event("neg_x", x < 0)
    event("in_box", ret == 1)
    return ret

def post(Pr):
    num = Pr({"in_box" : True, "neg_x" : True}) / Pr({"neg_x" : True})
    den = Pr({"in_box" : True, "neg_x" : False}) / Pr({"neg_x" : False})
    ratio = num / den
    return ratio > 0.95
'''

p = parse_fr(prog_string)

Holes = namedtuple('Holes', p.hole_defaults[0])
H_default = Holes(**p.hole_defaults[1])

repair_model = SMTRepair(p.D_exec, p.D_z3, p.z3_vars.inputs, p.z3_vars.output, Holes)

orig_prog = p.D_exec.partial_evaluate(*H_default)

evaluator = SamplingEvaluator(Sampler(p.pre_exec), p.post_exec, orig_prog)

d = Digits(p.pre_exec, orig_prog, repair_model, evaluator, max_depth=10)
soln_gen = d.soln_gen()

best = None
while True:
    try:
        n = next(soln_gen)
    except StopIteration:
        break
    print(n.path, ":", "(" + str(n.solution.post) + "," + str(n.solution.error) + ")" \
                       if n.solution is not None else str(None))
    if n.solution is not None and n.solution.post:
        if best is None or best.solution.error > n.solution.error:
            best = n

soln = best.solution
print("best repair:", best.path)
print("holes", [float(soln.holes[i]) for i in range(len(soln.holes))])
print("error", soln.error)
print("stats:", repair_model.stats)
