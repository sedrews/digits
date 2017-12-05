from abc import ABC, abstractmethod


# Using digits requires providing an implementation of
# 1) Evaluator -- a way to check the postcondition and compute the error function
# 2) RepairModel -- turn constraints into consistent programs


class Evaluator(ABC):

    @abstractmethod
    def compute_post(self, prog):
        pass

    @abstractmethod
    def compute_error(self, prog):
        pass


class RepairModel(ABC):

    # Can Override this to do something more sophisticated
    def initial_solution(self, program):
        return Solution(prog=program, post=False, error=0)

    # constraints is a dictionary from the input tuple to the output value
    @abstractmethod
    def make_solution(self, constraints):
        pass


# Those implementations of Evaluator and RepairModel
# must use some common classes


# You can subclass this to add more functionality
class Solution(object):

    # There will be potentially very many of these objects in memory
    __slots__ = 'prog','post','error'

    def __init__(self, prog=None, post=None, error=None):
        self.prog = prog # An executable function; digits calls it with a single arg from Sampler
        self.post = post
        self.error = error


# Structures for Digits to use internally


# Right now this badly emulates a list, but eventually it will be sophisticated
class _Queue:

    def __init__(self):
        self.elements = []

    def __len__(self):
        return len(self.elements)

    def pop(self):
        ret = self.elements[0]
        del self.elements[0]
        return ret

    def append(self, element):
        self.elements.append(element)


# I don't actually know what to do with this yet
class Sampler:

    def __init__(self, prob_prog):
        self.prob_prog = prob_prog

    # This should return a named tuple
    def next_sample(self):
        return self.prob_prog()


# Digits itself


class Digits:
    
    # Precondition can either be an function that returns a named tuple
    # or an instance of Sampler
    def __init__(self, precondition, repair_model, outputs=[0,1]):
        self.repair_model = repair_model
        if isinstance(precondition, Sampler):
            self.sampler = precondition
        else:
            self.sampler = Sampler(precondition)
        self.outputs = outputs

    def repair(self, n, program, evaluator):
        samples = []

        # Can encode the tree as a map from the constraints to solutions
        solutions = {}
        solutions[()] = self.repair_model.initial_solution(program)

        # For determining what solutions remain to be computed
        leaves = [()] # Initally just the empty tuple; eventually stuff like (0,1,1,0) (0,1,1,1) etc
        worklist = _Queue()

        while len(samples) < n:
            samples.append(self.sampler.next_sample())
            print("starting depth", len(samples), "to split", len(leaves), "leaves")
            for leaf in leaves:
                worklist.append(leaf)
            while not len(worklist) == 0:
                leaf = worklist.pop()
                leaves.remove(leaf)
                # Explore this leaf's children
                # Run the program at this leaf to propagate its solution to one child
                val = solutions[leaf].prog(samples[-1])
                for value in self.outputs:
                    if value == val: # We can use the same solution object
                        child = solutions[leaf]
                        leaves.append((*leaf, value))
                        solutions[(*leaf, value)] = child
                    else: # We have to compute the solution
                        child = self.repair_model.make_solution(dict(zip(samples, (*leaf, value))))
                        if child is not None:
                            leaves.append((*leaf, value))
                            solutions[(*leaf, value)] = child
                            child.post = evaluator.compute_post(child.prog)
                            if child.post: # Only compute error for correct solutions
                                child.error = evaluator.compute_error(child.prog)
        # TODO .values() is inefficient with multiplicity
        #return min([soln for soln in solutions.values() if soln.post], key=lambda x : x.error)
        return solutions.values()
