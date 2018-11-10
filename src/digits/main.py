from abc import ABC, abstractmethod
from functools import total_ordering, reduce
import heapq

import digits.tracking as tracking

# Using digits requires providing an implementation of
#   1) Evaluator -- a way to check the postcondition and compute the error function
#   2) RepairModel -- turn constraints into consistent programs
# (see smtrepair and samplingevaluator)


class Evaluator(ABC):

    @abstractmethod
    def compute_post(self, prog, fast=True):
        pass

    @abstractmethod
    def compute_error(self, prog, fast=True):
        pass

    @property
    def stats(self):
        return None


class RepairModel(ABC):

    # Can override this to do something more sophisticated
    def initial_solution(self, program):
        return Solution(prog=program, post=False, error=0)

    # constraints is a list of (input,output) tuples;
    # guarantee the convention that their order matches the sampleset
    @abstractmethod
    def make_solution(self, constraints):
        pass

    @property
    def stats(self):
        return None


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


class Frontier:
    
    def __init__(self, parent_ref):
        self._parent_ref = parent_ref
        self.items = list()

    @property
    def valuation(self):
        return lambda n : hamming_count(n.path, self._parent_ref.original_labeling)

    @property
    def threshold(self):
        return self._parent_ref.hthresh * self._parent_ref.depth

    @property
    def depth(self):
        return self._parent_ref.depth

    # Respects changes to parameters mid-generator
    def unblocked_generator(self):
        i = 0
        while True:
            if i >= len(self.items):
                break
            item = self.items[i]
            if len(item.path) <= self.depth and self.valuation(item) <= self.threshold:
                del self.items[i]
                i -= 1
                yield item
            i += 1
        self._active_gen = False


class Node:
    __slots__ = 'path','solution','parent','children','propto'
    def __init__(self, **kwargs):
        for key in self.__slots__:
            setattr(self, key, kwargs[key] if key in kwargs else None)
        if self.children is None:
            self.children = {}


def hamming_count(ell1, ell2):
    return len([t for t in zip(ell1, ell2) if t[0] != t[1]])


#
# Digits itself
#


class Digits:

    def __init__(self, precondition, program, repair_model, evaluator, outputs=(0,1), max_depth=None, hthresh=1, adaptive=None):

        if isinstance(precondition, Sampler):
            self.sampler = precondition
        else:
            self.sampler = Sampler(precondition)
        self.original_program = program
        self.repair_model = repair_model
        self.evaluator = evaluator
        self.outputs = outputs

        # We only consider constraint strings with at most this length (inclusive)
        self.max_depth = max_depth
        # Hamming distance heuristic threshold (default 1 => level-order traversal)
        self.hthresh = hthresh
        # Whether to change self.hthresh dynamically
        self.adaptive = adaptive

    def _initialize_search(self):
        self._best = None

        # The original program forms the root of the tree
        self.root = Node(path=(),solution=self.repair_model.initial_solution(self.original_program))

        self.depth = 1 # Bounds the largest constraint string explored (dynamically increases)

        assert self.max_depth is not None # XXX reproducing random seed results is hard if future samples depend on what has happened during the search
        self.original_labeling = [self.original_program(*self.sampler.get(i)) for i in range(self.max_depth)]
        #print("orig labeling:", self.original_labeling)

        # frontier contains (yet-unexplored) children of existing leaves
        self.frontier = Frontier(self)
        self._add_children(self.root)

    @property
    def best(self):
        return self._best

    def soln_gen(self):
        self._initialize_search()

        tracking.start_timer()
        tracking.log_event("entered generator")

        # XXX Won't terminate (but should) if all leaves are unsat
        while True:
            for leaf in self.frontier.unblocked_generator():
                tracking.log_event("popped leaf",
                                   length=len(leaf.path),
                                   valuation=self.frontier.valuation(leaf),
                                   path=reduce(lambda x,y : str(x) + str(y), leaf.path))
                if self._check_solution_propagation(leaf): # We can propagate the solution
                    leaf.solution = leaf.parent.solution
                    self._add_children(leaf)
                else: # We have to compute a solution (if one exists)
                    constraints = self._constraint_list(leaf.path)
                    leaf.solution = self.repair_model.make_solution(constraints)
                    if leaf.solution is not None:
                        # Handle computing error, updating best, etc
                        self._process_solution(leaf)
                        self._add_children(leaf)
                # Always yield what was done at this round
                yield leaf

            tracking.log_event("finished depth", depth=self.depth)
            tracking.log_stats("synthesizer", self.repair_model.stats)
            tracking.log_stats("evaluator", self.evaluator.stats)
            # We need to expand the depth of the search now that all unblocked are exhausted
            self.depth += 1
            if self.depth > self.max_depth:
                break

        tracking.log_event("exhausted generator")

    def _check_solution_propagation(self, leaf):
        parent = leaf.parent
        if parent.propto is None:
            # Run the parent program to see which child receives propagation (and cache)
            val = parent.solution.prog(*self.sampler.get(len(leaf.path) - 1))
            parent.propto = val
        return leaf.path[-1] == parent.propto

    def _process_solution(self, leaf):
        assert leaf.solution is not None
        self._evaluate(leaf, fast=True)
        if self._check_best(leaf): # If it looks like we should update the best solution
            # First be more precise
            self._evaluate(leaf, fast=False)
            if self._check_best(leaf): # If it still looks like it, do so
                self._best = leaf
                tracking.log_event("new best", error=self._best.solution.error)
                if self.adaptive is not None:
                    # Update the search threshold
                    self._update_hthresh(leaf.solution.error)


    def _evaluate(self, leaf, fast):
        leaf.solution.post = self.evaluator.compute_post(leaf.solution.prog, fast)
        if leaf.solution.post: # Only compute error for correct solutions
            leaf.solution.error = self.evaluator.compute_error(leaf.solution.prog, fast)

    def _check_best(self, leaf):
        if leaf.solution.post:
            if self._best is None or leaf.solution.error < self._best.solution.error:
                return True
        return False
            
    def _update_hthresh(self, error):
        new_thresh = self.adaptive[0] * error + self.adaptive[1]
        if new_thresh < self.hthresh:
            tracking.log_event("updated threshold", threshold=new_thresh)
            self.hthresh = new_thresh

    def _add_children(self, parent):
        # Only add the children if they are at a depth we would consider
        if self.max_depth is None or len(parent.path) < self.max_depth: # So len(child.path) <= max
            for value in self.outputs:
                child = Node(path=parent.path+(value,), parent=parent)
                parent.children[value] = child
                self.frontier.items.append(child)

    def _constraint_list(self, path):
        samples = [self.sampler.get(n) for n in range(len(path))]
        return list(zip(samples, path))
