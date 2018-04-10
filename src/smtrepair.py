from digits import *
from z3 import *
from parse import EventFunc


class SMTSolution(Solution):
    __slots__ = 'holes'
    def __init__(self, prog=None, post=None, error=None, holes=None):
        super().__init__(prog, post, error)
        self.holes = holes


class SMTRepair(RepairModel):

    def __init__(self, sketch, template, input_variables, output_variable, Holes):
        self.sketch = sketch
        self.template = template
        self.input_variables = input_variables
        self.output_variable = output_variable
        self.Holes = Holes

        self.unsat_core_list = []

    # constraints maps each input Sample to an ouput
    def make_solution(self, constraints):
        # Try unsat core pruning
        if self.core_matches(constraints):
            return None

        # Do actual synthesis work
        s = Solver()
        conj_ids = {} # Map conjunctions (i.e. (Sample,output) pairs) to their ids for unsat cores
        for constraint in constraints.items():
            #s.add(self.constraint_to_z3(constraint))
            conj_id = 'p' + str(len(conj_ids)) # XXX this could collide with variable names
            s.assert_and_track(self.constraint_to_z3(constraint), conj_id)
            conj_ids[constraint] = conj_id
        if s.check() == z3.sat:
            # Build and return a Solution instance
            hole_values = self.holes_from_model(s.model()) # Do not inline this below
            #print("made soln with holes", self.Holes(*[float(val) for val in hole_values]))
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
        else: #unsat
            # Extract an unsat core (when non-trivial)
            if len(s.unsat_core()) < len(constraints):
                core = [str(v) for v in s.unsat_core()]
                # If core contains the id for constraint c, we want to include it in our core list
                d = dict([constraint for constraint in constraints.items() if conj_ids[constraint] in core])
                self.unsat_core_list.append(d)
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
        for core in self.unsat_core_list:
            match = True
            for key in core:
                if core[key] != constraints[key]:
                    match = False
                    break
            if match:
                return True
        return False

    def sanity_check(self, soln, constraints):
        for sample,output in constraints.items():
            o = soln(*sample)
            assert o == output, str(sample) + ' does not map to ' + str(output) + '; instead ' + str(o)
