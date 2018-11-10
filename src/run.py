import argparse
from collections import namedtuple
import random
import sys
import time

import numpy

from digits import *


def run_benchmark(filename, max_depth, timeout, random_seed, eval_sample_size, opt_ratio, adapt, fspec=None):

    start = time.time()

    if random_seed is not None:
        random.seed(random_seed)
        numpy.random.seed(random_seed)

    f = open(filename, "r")
    prog_string = f.read()
    f.close()
    p = parse_fr(prog_string)

    Holes = namedtuple('Holes', p.hole_data[0])
    H_default = Holes(**{hole : p.hole_data[1][hole].default for hole in p.hole_data[0]})
    H_bounds = {hole : p.hole_data[1][hole].bounds for hole in p.hole_data[0]}

    repair_model = SMTRepair(p.D_exec, p.D_z3, p.z3_vars.inputs, p.z3_vars.output, Holes, H_bounds)

    orig_prog = p.D_exec.partial_evaluate(*H_default)
    
    s1 = Sampler(p.pre_exec)
    s2 = Sampler(p.pre_exec)
    # Prefetch all samples (controls randomness when search order changes)
    s1.get(max_depth)
    s2.get(eval_sample_size[1])

    evaluator = SamplingEvaluator(s2, p.post_exec, orig_prog if fspec is None else fspec, eval_sample_size)

    print("initial overhead,",time.time() - start)

    d = Digits(s1, orig_prog, repair_model, evaluator, max_depth=max_depth, hthresh=opt_ratio, adaptive=adapt, fspec=fspec)
    soln_gen = d.soln_gen()

    start = time.time()
    while True:
        try:
            n = next(soln_gen)
        except StopIteration:
            break
        if timeout is not None and time.time() - start > timeout:
            print("timed out")
            break
    if d.best is not None:
        print("best:")
        print("  error", d.best.solution.error)
        print("  holes", [float(hole) for hole in d.best.solution.holes])
    else:
        print("no solutions found")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, type=str,
                        help='Input .fr file')
    parser.add_argument('-d', '--depth', required=True, type=int,
                        help='Maximum depth of the search')
    parser.add_argument('-t', '--time', required=False, type=int, default=None,
                        help='Timeout (seconds) for the search')
    parser.add_argument('-s', '--seed', required=False, type=int, default=None,
                        help='Set the random seed')
    parser.add_argument('-sz', '--size', required=False, nargs=2, type=int, default=(10000,73778),
                        help='The number of samples used in the sampling-based evaluator as a tuple (fast,full)')
    parser.add_argument('-o', '--opt', required=False, type=float, default=1,
                        help='The ratio of the depth used as a Hamming distance threshold for the optimized search; when == 1, the search proceeds in level-order')
    parser.add_argument('-a', '--adapt', required=False, nargs=2, type=float, default=None,
                        help='If specified, let (a,b) be its value: updates the --opt value to ae+b when finding a correct solution with error e')
    parser.add_argument('-fs', '--fspec', required=False, type=str, default=None,
                        help='Overrides repair: write a lambda function as a string to be eval()')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    fspec = None if args.fspec is None else eval(args.fspec)
    run_benchmark(args.file, args.depth, args.time, args.seed, args.size, args.opt, args.adapt, fspec)
