from functools import reduce
from itertools import chain
import time

from gmpy2 import mpq
from z3 import *

from .main import *
from .tracking import Stats
from .cvc4 import cvc4_query_from_z3


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
    def __init__(self, sketch, template, input_variables, output_variable, Holes, hole_bounds=None, cvc4=False):
        self.sketch = sketch
        self.template = template
        self.input_variables = input_variables
        self.output_variable = output_variable
        self.Holes = Holes
        self.hole_bounds = hole_bounds
        self.cvc4 = cvc4

        self.unsat_cores = CoreStore()
        self._stats = Stats(calls = 0, # calls = smt + pruned
                            smt = 0, # smt = sat + unsat
                            sat = 0, unsat = 0,
                            pruned = 0, # Pruned using unsat cores
                            make_time = 0, # Includes sanity_time
                            sanity_time = 0)

    @property
    def stats(self):
        return self._stats

    # constraints maps each input Sample to an ouput (stored as a list of tuples) --
    # Digits maintains a guarantee about their fixed ordering across multiple calls
    def make_solution(self, constraints):
        start_make_time = time.time()
        self.stats.calls += 1

        # Try unsat core pruning
        if self.unsat_cores.check_match([c[1] for c in constraints]):
            self.stats.pruned += 1
            #s = Solver()
            #for i in range(len(constraints)):
            #    s.add(self.constraint_to_z3(constraints[i]))
            #assert s.check() == unsat
            return None

        # Do actual synthesis work
        s = self._build_Solver(constraints)
        res = self._smt_query(s, constraints)
        self.stats.make_time += time.time() - start_make_time
        return res

    def _smt_query(self, s, constraints):
        self.stats.smt += 1
        if not self.cvc4: # Use z3
            if s.check() == z3.sat:
                self.stats.sat += 1
                # Build and return a Solution instance
                hole_values = self.holes_from_model(s.model())
                soln = self.sketch.partial_evaluate(*hole_values)
                # Make sure the synthesized program is consistent with constraints
                self.sanity_check(soln, constraints)
                return SMTSolution(prog=soln, holes=hole_values)
            else: # unsat
                self.stats.unsat += 1
                self.extract_unsat_core(s.unsat_core(), constraints)
                return None
        else: # Use cvc4
            model,core = cvc4_query_from_z3(s, list(self.Holes._fields), ['p' + str(i) for i in range(len(constraints))])
            if model is not None: # sat
                assert core is None
                self.stats.sat += 1
                hole_values = self.Holes(**{attr : mpq(*model[attr]) for attr in model})
                soln = self.sketch.partial_evaluate(*hole_values)
                # Make sure the synthesized program is consistent with constraints
                self.sanity_check(soln, constraints)
                return SMTSolution(prog=soln, holes=hole_values)
            elif core is not None: # unsat
                assert model is None
                self.stats.unsat += 1
                self.extract_unsat_core(core, constraints)
                return None
            else:
                assert False

    def extract_unsat_core(self, core, constraints):
        if len(core) < len(constraints): # If the core is non-trivial
            # The names of the variables stored their constraint index,
            # i.e. some str(v) below looks like 'p15' when constraints[15] contributes to the unsat core
            core = sorted([int(str(v)[1:]) for v in core])
            # Represent as a list of (constraint number, specified output)
            core = [(v,constraints[v][1]) for v in core]
            self.unsat_cores.add_core(core)

    def _build_Solver(self, constraints):
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
        return s

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
        start_sanity_time = time.time()
        self._sanity_check(soln, constraints)
        self.stats.sanity_time += time.time() - start_sanity_time

    def _sanity_check(self, soln, constraints):
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
