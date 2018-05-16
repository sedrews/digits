import argparse
from collections import namedtuple
import random
import sys

import numpy

from digits import *


def run_benchmark(filename, max_depth, random_seed, eval_sample_size, opt_ratio, adapt):
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
        s2.get(eval_sample_size[1])

    evaluator = SamplingEvaluator(s2, p.post_exec, orig_prog, fast_num=eval_sample_size[0], num=eval_sample_size[1])

    d = Digits(s1, orig_prog, repair_model, evaluator, max_depth=max_depth, hthresh=opt_ratio, adaptive=adapt)
    soln_gen = d.soln_gen()

    while True:
        try:
            n = next(soln_gen)
        except StopIteration:
            break
    d.log_event("best holes", *[float(hole) for hole in d.best.solution.holes])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, type=str,
                        help='Input .fr file')
    parser.add_argument('-d', '--depth', required=True, type=int,
                        help='Maximum depth of the search')
    parser.add_argument('-s', '--seed', required=False, type=int, default=None,
                        help='Set the random seed')
    parser.add_argument('-sz', '--size', required=False, nargs=2, type=int, default=(10000,73778),
                        help='The number of samples used in the sampling-based evaluator as a tuple (fast,full)')
    parser.add_argument('-o', '--opt', required=False, type=float, default=1,
                        help='The ratio of the depth used as a Hamming distance threshold for the optimized search; when == 1, the search proceeds in level-order')
    parser.add_argument('-a', '--adapt', required=False, nargs=2, type=float, default=None,
                        help='If specified, let (a,b) be its value: updates the --opt value to ae+b when finding a correct solution with error e')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_benchmark(args.file, args.depth, args.seed, args.size, args.opt, args.adapt)
