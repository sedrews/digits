from digits import *
from z3 import *


class SMTRepair(RepairModel):

    def __init__(self, sketch, template, output_variable, Holes):
        self.sketch = sketch
        self.template = template
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
        for constraint in constraints.items():
            s.add(self.constraint_to_z3(constraint))
        if s.check() == z3.sat:
            # Build and return a Solution instance
            hole_values = self.holes_from_model(s.model()) # Do not inline this below
            soln = lambda inputs : self.sketch(inputs, hole_values)
            return Solution(prog=soln)
        else: #unsat
            # Extract an unsat core TODO
            return None

    # constraint is a tuple of (Sample, output) where Sample is a named input tuple
    def constraint_to_z3(self, constraint):
        exp_pairs = [(Real(attr), RealVal(getattr(constraint[0], attr))) for attr in constraint[0]._fields]
        exp_pairs += [(self.output_variable,RealVal(constraint[1]))]
        return substitute(self.template, exp_pairs)

    def holes_from_model(self, model):
        return self.Holes(*[model[Real(attr)].as_fraction() for attr in self.Holes._fields])

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
