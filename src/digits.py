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
        self.prog = prog # Anything implementing __call__ (see parse.EventFunc)
        self.post = post
        self.error = error


# Structures for Digits to use internally


# Right now this badly emulates a list, but eventually it will be sophisticated
class _Queue:

    def __init__(self):
        self.items = []

    def qsize(self):
        return len(self.items)

    def get(self):
        ret = self.items[0]
        del self.items[0]
        return ret

    def put(self, item):
        self.items.append(item)


class Sampler:

    def __init__(self, prob_prog):
        assert callable(prob_prog)
        self.prob_prog = prob_prog
        self.samples = []

    # Returns the n-th sample (convention: as a tuple; must be unpacked when used as prog input)
    def get(self, n):
        while n >= len(self.samples):
            self.samples.append(self.prob_prog())
        return self.samples[n]


class Node:
    __slots__ = 'path','solution','parent','children','propto'
    def __init__(self, **kwargs):
        for key in self.__slots__:
            setattr(self, key, kwargs[key] if key in kwargs else None)
        if self.children is None:
            self.children = {}


# Digits itself (and auxiliary methods)

def digits(precondition, program, repair_model, evaluator, outputs=[0,1], n=10):
    if isinstance(precondition, Sampler):
        sampler = precondition
    else:
        sampler = Sampler(precondition)

    # The original program forms the root of the tree
    root = Node(path=(),solution=repair_model.initial_solution(program))
    worklist = _Queue() # Should contain (yet-unexplored) children of existing leaves
    add_children(worklist, root, outputs)

    # The main loop
    while worklist.qsize() > 0:

        leaf = worklist.get()

        # Handle solution propagation
        parent = leaf.parent
        if parent.propto is None:
            # Run the parent program to see which child receives propagation
            val = parent.solution.prog(*sampler.get(len(leaf.path) - 1))

        if leaf.path[-1] == parent.propto: # We can propagate the solution
            leaf.solution = parent.solution
            add_children(worklist, leaf, outputs)
        else: # We have to compute a solution (if one exists)
            leaf.solution = repair_model.make_solution(constraint_dict(sampler, leaf.path))
            if leaf.solution is not None:
                add_children(worklist, leaf, outputs)
                leaf.solution.post = evaluator.compute_post(leaf.solution.prog)
                if leaf.solution.post: # Only compute error for correct solutions
                    leaf.solution.error = evaluator.compute_error(leaf.solution.prog)
        
        yield leaf

def add_children(worklist, parent, outputs):
    for value in outputs:
        child = Node(path=parent.path+(value,), parent=parent)
        parent.children[value] = child
        worklist.put(child)

def constraint_dict(sampler, path):
    samples = [sampler.get(n) for n in range(len(path))]
    return dict(zip(samples, path))
