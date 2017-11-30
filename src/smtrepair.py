from digits import *


class SMTRepair(RepairModel):

    def __init__(self):
        self.unsat_core_list = []

    def make_solution(self, constraints):
        # Try unsat core pruning
        if self.core_matches(constraints):
            return None
        
        # Do actual synthesis work
        pass

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
