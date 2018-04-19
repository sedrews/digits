from parse import parse_fr
from digits import Digits, Sampler
from smtrepair import SMTRepair
from samplingevaluator import SamplingEvaluator

from collections import namedtuple

import sys
import argparse

import random
import numpy

def run_benchmark(filename, max_depth, random_seed):
    if random_seed is not None:
        random.seed(random_seed)
        numpy.random.seed(random_seed)

    f = open(filename, "r")
    prog_string = f.read()
    f.close()
    p = parse_fr(prog_string)

    Holes = namedtuple('Holes', p.hole_defaults[0])
    H_default = Holes(**p.hole_defaults[1])

    repair_model = SMTRepair(p.D_exec, p.D_z3, p.z3_vars.inputs, p.z3_vars.output, Holes)

    orig_prog = p.D_exec.partial_evaluate(*H_default)
    
    s1 = Sampler(p.pre_exec)
    s2 = Sampler(p.pre_exec)
    # If we're controlling for random seed, prefetch all samples
    if random_seed is not None:
        s1.get(max_depth)
        s2.get(2000)

    evaluator = SamplingEvaluator(s2, p.post_exec, orig_prog)

    d = Digits(s1, orig_prog, repair_model, evaluator, max_depth=max_depth)
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, type=str,
                        help='Input .fr file')
    parser.add_argument('-d', '--depth', required=True, type=int,
                        help='Maximum depth of the search')
    parser.add_argument('-s', '--seed', required=False, type=int, default=None,
                        help='Set the random seed')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_benchmark(args.file, args.depth, args.seed)
