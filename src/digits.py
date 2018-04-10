from abc import ABC, abstractmethod


# Using digits requires providing an implementation of
#   1) Evaluator -- a way to check the postcondition and compute the error function
#   2) RepairModel -- turn constraints into consistent programs
# (see smtrepair and samplingevaluator)


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


# Implementations of Evaluator and RepairModel need a common Solution class
# (can be subclassed for additional functionality)
class Solution(object):

    # There will be potentially very many of these objects in memory
    __slots__ = 'prog','post','error'

    def __init__(self, prog=None, post=None, error=None):
        self.prog = prog # Anything implementing callable (see parse.EventFunc)
        self.post = post
        self.error = error


#
# Structures for Digits to use internally
#


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


#
# Digits itself
#


class Digits:

    def __init__(self, precondition, program, repair_model, evaluator, outputs=(0,1)):
        if isinstance(precondition, Sampler):
            self.sampler = precondition
        else:
            self.sampler = Sampler(precondition)
        self.original_program = program
        self.repair_model = repair_model
        self.evaluator = evaluator
        self.outputs = outputs
        
        # Maintain an explicit representation of the tree as a map from binary strings to nodes
        self.tree = {} # Note this will also include non-explored nodes in the worklist
        # The original program forms the root of the tree
        self.root = Node(path=(),solution=repair_model.initial_solution(program))
        self.tree[()] = self.root

        # worklist contains (yet-unexplored) children of existing leaves
        self.worklist = _Queue()
        self._add_children(self.root)


    def soln_gen(self):
        while self.worklist.qsize() > 0:

            leaf = self.worklist.get()

            # Handle solution propagation
            parent = leaf.parent
            if parent.propto is None:
                # Run the parent program to see which child receives propagation
                val = parent.solution.prog(*self.sampler.get(len(leaf.path) - 1))

            if leaf.path[-1] == parent.propto: # We can propagate the solution
                leaf.solution = parent.solution
                self._add_children(leaf)
            else: # We have to compute a solution (if one exists)
                constraints = self._constraint_dict(leaf.path)
                leaf.solution = self.repair_model.make_solution(constraints)
                if leaf.solution is not None:
                    self._add_children(leaf)
                    leaf.solution.post = self.evaluator.compute_post(leaf.solution.prog)
                    if leaf.solution.post: # Only compute error for correct solutions
                        leaf.solution.error = self.evaluator.compute_error(leaf.solution.prog)
            
            yield leaf

    def _add_children(self, parent):
        for value in self.outputs:
            child = Node(path=parent.path+(value,), parent=parent)
            parent.children[value] = child
            self.worklist.put(child)
            self.tree[child.path] = child

    def _constraint_dict(self, path):
        samples = [self.sampler.get(n) for n in range(len(path))]
        return dict(zip(samples, path))
