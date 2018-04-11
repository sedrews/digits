from digits import *
from z3 import *
from parse import EventFunc


class SMTSolution(Solution):
    __slots__ = 'holes'
    def __init__(self, prog=None, post=None, error=None, holes=None):
        super().__init__(prog, post, error)
        self.holes = holes


class SMTRepair(RepairModel):

    # sketch is a parse.EventFunc with some of the input variables being holes
    # template is a z3 formula representing program whose free variables are inputs,holes,output
    # input_variables is a list of the z3 variable objects (in the same order as in Sample and in function header)
    # output_variable is the z3 variable object for the return value
    # Holes is a named tuple whose fields are the hole variable names
    def __init__(self, sketch, template, input_variables, output_variable, Holes):
        self.sketch = sketch
        self.template = template
        self.input_variables = input_variables
        self.output_variable = output_variable
        self.Holes = Holes

        # A separate CoreStore for each depth (can prove matches never occur between different depths)
        self.unsat_cores = {}

    # constraints maps each input Sample to an ouput (stored as a list of tuples) --
    # Digits maintains a guarantee about their fixed ordering across multiple calls
    def make_solution(self, constraints):
        # Try unsat core pruning
        if self.core_matches(constraints):
            return None

        # Do actual synthesis work
        s = Solver()
        for i in range(len(constraints)):
            constraint = constraints[i]
            conj_id = 'p' + str(i) # XXX this could collide with variable names
            s.assert_and_track(self.constraint_to_z3(constraint), conj_id)
        if s.check() == z3.sat:
            # Build and return a Solution instance
            hole_values = self.holes_from_model(s.model())
            soln = self.sketch.partial_evaluate(*hole_values)
            try:
                # Make sure the synthesized program is consistent with constraints
                self.sanity_check(soln, constraints)
            except AssertionError as e:
                print("sanity check failed:")
                print(e)
                print("constraints", constraints)
                print("holes", hole_values)
                print("(float holes)", self.Holes(*[float(val) for val in hole_values]))
                print("solver", s)
                print("model", s.model())
                exit(1)
            return SMTSolution(prog=soln, holes=hole_values)
        else: # unsat
            # Extract an unsat core (when non-trivial)
            if len(s.unsat_core()) < len(constraints):
                # The names of the variables stored their constraint index,
                # i.e. some str(v) below looks like 'p15' when constraints[15] contributes to the unsat core
                core = [int(str(v)[1:]) for v in s.unsat_core()]
                # Represent as a list of (constraint number, specified output)
                core = [(v,constraints[v][1]) for v in core]
                self.core_process(core)
            return None

    # constraint is a tuple of (Sample, output) where Sample is an input tuple
    def constraint_to_z3(self, constraint):
        exp_pairs = [(self.input_variables[i], RealVal(constraint[0][i])) for i in range(len(constraint[0]))]
        exp_pairs += [(self.output_variable, RealVal(constraint[1]))]
        sub = substitute(self.template, exp_pairs)
        return sub

    def holes_from_model(self, model):
        #return self.Holes(*[model[Real(attr)].as_fraction() for attr in self.Holes._fields])
        return self.Holes(*[model.evaluate(Real(attr), model_completion=True).as_fraction() \
                            for attr in self.Holes._fields])

    def core_matches(self, constraints):
        coreindex = len(constraints) - 1
        if coreindex in self.unsat_cores:
            return self.unsat_cores[coreindex].check_match([c[0] for c in constraints])
        return False

    def core_process(self, core):
        assert(len(core) > 0)
        core.sort(key=lambda t : t[0], reverse=True)
        coreindex = core[0][0]
        if coreindex not in self.unsat_cores:
            self.unsat_cores[coreindex] = CoreStore()
        self.unsat_cores[coreindex].add_core(core)

    def sanity_check(self, soln, constraints):
        for sample,output in constraints:
            o = soln(*sample)
            assert o == output, str(sample) + ' does not map to ' + str(output) + '; instead ' + str(o)


class CoreStore:

    def __init__(self):
        self.core_list = []

    # constraints is a list of output values
    def check_match(self, constraints):
        for core in self.core_list:
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
        self.core_list.append(core)
