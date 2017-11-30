from digits import *


class SMTSolution(Solution):

    __slots__ = 'original_constraints','implicit_constraints'

    def __init__(self):
        super().__init__()
        self.original_constraints = []
        self.implicit_constraints = []


class SMTRepair(RepairModel):

    def __init__(self):
        self.unsat_core_list = []

    def initial_solution(self, program):
        ret = SMTSolution()
        ret.prog = program #XXX make this something meaningful?

    def make_solution(self, oldsoln, constraint):
        #try unsat core pruning
        
        #do actual synthesis work
        pass

