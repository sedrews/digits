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
        soln = Solution()
        soln.prog = program
        soln.post = False
        soln.error = 0

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

    def __init__(self):
        self.prog = None # An executable function; digits calls it with a single arg from Sampler
        self.post = False
        self.error = 0


# Structures for Digits to use internally


class _Queue:

    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def pop(self):
        ret = self.elements[0]
        del self.elements[0]
        return ret

    def push(self, element):
        self.elements.append(element)


# I don't actually know what to do with this yet
class Sampler:

    def __init__(self):
        pass

    # This should return a named tuple
    def next_sample(self):
        pass


# Digits itself


class Digits:
    
    def __init__(self):
        pass

    def repair(self, n, sampler, repair_model, evaluator):
        outputs = [True, False]
        samples = []

        # Can encode the tree as a map from the constraints to solutions
        solutions = {}
        solutions[()] = repair_model.initial_solution(program)

        # For determining what solutions remain to be computed
        leaves = [()] # Initally just the empty tuple; eventually stuff like (0,1,1,0) (0,1,1,1) etc
        worklist = _Queue()

        while len(samples) < n:
            samples.append(sampler.next_sample())
            for leaf in leaves:
                worklist.push(leaf)
            while not worklist.empty():
                leaf = worklist.pop()
                leaves.remove(leaf)
                # Explore this leaf's children
                # Run the program at this leaf to propagate its solution to one child
                val = solutions[leaf].prog(*samples[-1])
                for value in outputs:
                    if value == val: # We can use the same solution object
                        child = solutions[leaf]
                        self.leaves.append((*leaf, value))
                        self.solutions[(*leaf, value)] = child
                    else: # We have to compute the solution
                        child = repair_model.make_solution(dict(zip(samples, (*leaf, value))))
                        if child is not None:
                            self.leaves.append((*leaf, value))
                            self.solutions[(*leaf, value)] = child
                            child.post = evaluator.compute_post(child.prog)
                            child.error = evaluator.compute_error(child.prog)
        # TODO .values() is inefficient with multiplicity
        return min([soln for soln in solutions.values() if soln.post], key=lambda x : x.error)
