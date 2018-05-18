from abc import ABC, abstractmethod
from functools import total_ordering, reduce
import heapq
import time


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

    def get_stats(self):
        return []

class RepairModel(ABC):

    # Can override this to do something more sophisticated
    def initial_solution(self, program):
        return Solution(prog=program, post=False, error=0)

    # constraints is a list of (input,output) tuples;
    # guarantee the convention that their order matches the sampleset
    @abstractmethod
    def make_solution(self, constraints):
        pass

    def get_stats(self):
        return []


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


class _HeapQueue:

    # heapq.heappush(tuple) seems to sort tuples lexicographically
    # -- we want to sort only by valuation stored at [0] as otherwise all elements need a comparator
    @total_ordering
    class _Tuple(tuple):
        def __eq__(self, other):
            if not isinstance(other, _HeapQueue._Tuple):
                return NotImplemented
            return self[0].__eq__(other[0])
        def __lt__(self, other):
            if not isinstance(other, _HeapQueue._Tuple):
                return NotImplemented
            return self[0].__lt__(other[0])

    def __init__(self, depth=None, valuation=None, threshold=None):
        self.items = [] # Stored as (valuation, item) but wrapped in _Tuple
        self._depth = depth
        self.valuation = valuation # A function : node -> numeric
        self.threshold = threshold # A function : depth -> numeric
        self.standby = [] # Stored as (depth, item) but wrapped in _Tuple

    @property
    def depth(self):
        return self._depth
    @depth.setter
    def depth(self, value):
        assert self._depth <= value
        self._depth = value
        # When we expand the depth, some new nodes can be considered
        while len(self.standby) > 0 and self.standby[0][0] <= self._depth:
            self.put(heapq.heappop(self.standby)[1])


    def qsize(self):
        return len(self.items) + len(self.standby)

    # Returns the least item that passes the threshold (or None if none pass)
    def get(self):
        # Recall self.items[n] = (valuation, item)
        if len(self.items) > 0 and self.items[0][0][0] <= self.threshold(self._depth):
            ret = heapq.heappop(self.items)
            #print("popped item with valuation", ret[0])
            return ret[1]
        else:
            return None

    def put(self, item):
        d = len(item.path)
        if d <= self._depth:
            # Sorts by valuation, tie-breaker by depth (so parents are visited before their minimal child)
            heapq.heappush(self.items, self._Tuple(((self.valuation(item),d), item)))
        else:
            heapq.heappush(self.standby, self._Tuple((d, item))) # Sorts by depth


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

    def __init__(self, precondition, program, repair_model, evaluator, outputs=(0,1), max_depth=None, hthresh=1, adaptive=None):

        self.start_time = time.time()

        if isinstance(precondition, Sampler):
            self.sampler = precondition
        else:
            self.sampler = Sampler(precondition)
        self.original_program = program
        self.repair_model = repair_model
        self.evaluator = evaluator
        self.outputs = outputs
        
        self._best = None

        # Maintain an (auxiliary) explicit representation of the tree as a map from binary strings to nodes
        self.tree = {} # Note this will also include non-explored nodes in the worklist
        # The original program forms the root of the tree
        self.root = Node(path=(),solution=repair_model.initial_solution(program))
        self.tree[()] = self.root

        self.max_depth = max_depth # We only consider constraint strings with at most this length (inclusive)
        self.depth = 0 # Bounds the largest constraint string explored (dynamically increases)

        assert self.max_depth is not None # XXX reproducing random seed results is hard if future samples depend on what has happened during the search
        self.original_labeling = [self.original_program(*self.sampler.get(i)) for i in range(self.max_depth)]
        #print("orig labeling:", self.original_labeling)

        self.hthresh = hthresh # Hamming distance heuristic threshold (default 1 => level-order traversal)
        # Nodes are sorted by their Hamming distance from the original program
        hamming_count = lambda n : len([i for i in range(len(n.path)) if n.path[i] != self.original_labeling[i]])
        # worklist contains (yet-unexplored) children of existing leaves
        self.worklist = _HeapQueue(self.depth, hamming_count, self._get_threshold_func())
        self._add_children(self.root)

        self.adaptive = adaptive # Whether to change self.hthresh dynamically

    @property
    def best(self):
        return self._best

    def _get_threshold_func(self):
        return lambda d : self.hthresh * d # We use a fraction of the depth as the threshold

    def soln_gen(self):
        self.log_event("entered generator")
        # XXX Never terminates if worklist threshold always blocks some nodes and no max depth is set
        while True:

            leaf = self.worklist.get()
            if leaf is None: # We need to expand the depth of the search
                self.log_event("finished depth", self.depth)
                self.log_event("synthesizer stats", *self.repair_model.get_stats())
                self.log_event("evaluator stats", *self.evaluator.get_stats())
                if self.worklist.qsize() == 0: # We exhausted the search space, somehow
                    break
                self.depth += 1 # We handle comparing to self.max_depth in _add_children
                # Let heuristic search be incomplete
                if self.depth > self.max_depth:
                    break
                self.worklist.depth = self.depth
                continue

            # Handle solution propagation
            parent = leaf.parent
            if parent.propto is None:
                # Run the parent program to see which child receives propagation (and cache)
                val = parent.solution.prog(*self.sampler.get(len(leaf.path) - 1))
                parent.propto = val

            if leaf.path[-1] == parent.propto: # We can propagate the solution
                leaf.solution = parent.solution
                self._add_children(leaf)
            else: # We have to compute a solution (if one exists)
                constraints = self._constraint_list(leaf.path)
                leaf.solution = self.repair_model.make_solution(constraints)
                if leaf.solution is not None:
                    self._add_children(leaf)
                    self._evaluate(leaf, fast=True)
                    if self._check_best(leaf): # If it looks like we should update the best solution
                        # First be more precise
                        self._evaluate(leaf, fast=False)
                        if self._check_best(leaf): # If it still looks like it, do so
                            self._best = leaf
                            self.log_event("new best", self._best.solution.error, \
                                    "path length", len(self._best.path), \
                                    "valuation", self.worklist.valuation(self._best))
                            if self.adaptive is not None:
                                # Update the search threshold
                                self._update_hthresh(leaf.solution.error)

            # Always yield what was done at this round
            yield leaf

        self.log_event("exhausted generator")

    def _evaluate(self, leaf, fast):
        leaf.solution.post = self.evaluator.compute_post(leaf.solution.prog, fast)
        if leaf.solution.post: # Only compute error for correct solutions
            leaf.solution.error = self.evaluator.compute_error(leaf.solution.prog, fast)

    def _check_best(self, leaf):
        if leaf.solution.post:
            if self._best is None or leaf.solution.error < self._best.solution.error:
                return True
        return False
            
    def log_event(self, *args):
        for arg in args:
            assert "," not in str(arg)
        print(reduce(lambda x,y : str(x) + "," + str(y), [time.time()-self.start_time] + list(args)))

    def _update_hthresh(self, error):
        new_thresh = self.adaptive[0] * error + self.adaptive[1]
        if new_thresh < self.hthresh:
            self.log_event("updated thresh", new_thresh)
            self.hthresh = new_thresh
            self.worklist.threshold = self._get_threshold_func()

    def _add_children(self, parent):
        # Only add the children if they are at a depth we would consider
        if self.max_depth is None or len(parent.path) < self.max_depth: # So len(child.path) <= max
            for value in self.outputs:
                child = Node(path=parent.path+(value,), parent=parent)
                parent.children[value] = child
                self.worklist.put(child)
                self.tree[child.path] = child

    def _constraint_list(self, path):
        samples = [self.sampler.get(n) for n in range(len(path))]
        return list(zip(samples, path))
