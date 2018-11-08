from functools import reduce
from itertools import chain
import time

from gmpy2 import mpq
from z3 import *

from .main import *


class SMTSolution(Solution):
    __slots__ = 'holes'
    def __init__(self, prog=None, post=None, error=None, holes=None):
        super().__init__(prog, post, error)
        self.holes = holes


class SMTRepair(RepairModel):

    # sketch is a parse.EventFunc with some of the input variables being holes -- it needs to have .partial_evaluate
    # template is a z3 formula representing program whose free variables are inputs,holes,output
    # input_variables is a list of the z3 variable objects (in the same order as in Sample and in function header)
    # output_variable is the z3 variable object for the return value
    # Holes is a named tuple whose fields are the hole variable names
    def __init__(self, sketch, template, input_variables, output_variable, Holes, hole_bounds=None):
        self.sketch = sketch
        self.template = template
        self.input_variables = input_variables
        self.output_variable = output_variable
        self.Holes = Holes
        self.hole_bounds = hole_bounds

        self.unsat_cores = CoreStore()
        class Stats:
            def __init__(self):
                self.calls = 0 # calls = smt + pruned
                self.smt = 0 # smt = sat + unsat
                self.sat = 0
                self.unsat = 0
                self.pruned = 0
                self.make_time = 0 # Includes sanity_time
                self.sanity_time = 0
                self._total_constraints = 0
            @property
            def avg_num_constraints(self):
                try:
                    return self._total_constraints / self.calls
                except ZeroDivisionError:
                    return 0
            def as_array(self):
                return [("calls", self.calls), \
                        ("smt", self.smt), \
                        ("sat", self.sat), \
                        ("unsat", self.unsat), \
                        ("pruned", self.pruned), \
                        ("make_time", self.make_time), \
                        ("sanity_time", self.sanity_time), \
                        ("avg_num_constraints", self.avg_num_constraints)]
            def __str__(self):
                return reduce(lambda x,y: x + y, [p[0]+":"+str(p[1]) for p in self.as_array()])
        self.stats = Stats()

    def get_stats(self):
        return [str(i) for i in chain(*self.stats.as_array())]

    # constraints maps each input Sample to an ouput (stored as a list of tuples) --
    # Digits maintains a guarantee about their fixed ordering across multiple calls
    def make_solution(self, constraints):
        start_make_time = time.time()
        self.stats.calls += 1
        self.stats._total_constraints += len(constraints)
        # Try unsat core pruning
        if self.unsat_cores.check_match([c[1] for c in constraints]):
            self.stats.pruned += 1
            #s = Solver()
            #for i in range(len(constraints)):
            #    s.add(self.constraint_to_z3(constraints[i]))
            #assert s.check() == unsat
            return None

        # Do actual synthesis work
        self.stats.smt += 1
        s = Solver()
        if self.hole_bounds is not None:
            for hole in self.Holes._fields:
                if self.hole_bounds[hole] is not None:
                    lb = self.hole_bounds[hole][0]
                    ub = self.hole_bounds[hole][1]
                    if lb is not None:
                        s.add(Real(hole) >= lb)
                    if ub is not None:
                        s.add(Real(hole) <= ub)
        for i in range(len(constraints)):
            constraint = constraints[i]
            conj_id = 'p' + str(i) # XXX this could collide with variable names
            s.assert_and_track(self.constraint_to_z3(constraint), conj_id)
        if s.check() == z3.sat:
            self.stats.sat += 1
            # Build and return a Solution instance
            hole_values = self.holes_from_model(s.model())
            soln = self.sketch.partial_evaluate(*hole_values)
            try:
                # Make sure the synthesized program is consistent with constraints
                start_sanity_time = time.time()
                self.sanity_check(soln, constraints)
                self.stats.sanity_time += time.time() - start_sanity_time
            except AssertionError as e:
                print("sanity check failed:")
                print(e)
                print("constraints", constraints)
                print("holes", hole_values)
                print("(float holes)", self.Holes(*[float(val) for val in hole_values]))
                print("solver", s)
                print("model", s.model())
                exit(1)
            self.stats.make_time += time.time() - start_make_time
            return SMTSolution(prog=soln, holes=hole_values)
        else: # unsat
            self.stats.unsat += 1
            # Extract an unsat core (when non-trivial)
            if len(s.unsat_core()) < len(constraints):
                # The names of the variables stored their constraint index,
                # i.e. some str(v) below looks like 'p15' when constraints[15] contributes to the unsat core
                core = sorted([int(str(v)[1:]) for v in s.unsat_core()])
                # Represent as a list of (constraint number, specified output)
                core = [(v,constraints[v][1]) for v in core]
                self.unsat_cores.add_core(core)
            self.stats.make_time += time.time() - start_make_time
            return None

    # constraint is a tuple of (Sample, output) where Sample is an input tuple
    def constraint_to_z3(self, constraint):
        exp_pairs = [(self.input_variables[i], RealVal(constraint[0][i])) for i in range(len(constraint[0]))]
        exp_pairs += [(self.output_variable, RealVal(constraint[1]))]
        sub = substitute(self.template, exp_pairs)
        return sub

    def holes_from_model(self, model):
        #return self.Holes(*[model[Real(attr)].as_fraction() for attr in self.Holes._fields])
        return self.Holes(*[mpq(model.evaluate(Real(attr), model_completion=True).as_fraction()) \
                            for attr in self.Holes._fields])

    def sanity_check(self, soln, constraints):
        for sample,output in constraints:
            o = soln(*sample)
            assert o == output, str(sample) + ' does not map to ' + str(output) + '; instead ' + str(o)


class CoreStore:

    def __init__(self):
        self.core_lists = {} # Separate list for each depth

    def _fetch(self, n):
        if n not in self.core_lists:
            self.core_lists[n] = []
        return self.core_lists[n]

    # constraints is a list of output values
    def check_match(self, constraints):
        for core in self._fetch(len(constraints) - 1):
            match = True
            for index,out in core:
                if constraints[index] != out:
                    match = False
                    break
            if match:
                return True
        return False

    # core is a list of (index of constraint, specified output value)
    def add_core(self, core):
        assert len(core) > 0
        self._fetch(max([c[0] for c in core])).append(core)
