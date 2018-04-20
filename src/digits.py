from abc import ABC, abstractmethod
import heapq
from functools import total_ordering


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

    # Can override this to do something more sophisticated
    def initial_solution(self, program):
        return Solution(prog=program, post=False, error=0)

    # constraints is a list of (input,output) tuples;
    # guarantee the convention that their order matches the sampleset
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
        if len(self.items) > 0 and self.items[0][0] <= self.threshold(self._depth):
            ret = heapq.heappop(self.items)
            #print("popped item with valuation", ret[0])
            return ret[1]
        else:
            return None

    def put(self, item):
        d = len(item.path)
        if d <= self._depth:
            heapq.heappush(self.items, self._Tuple((self.valuation(item), item))) # Sorts by valuation
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

    def __init__(self, precondition, program, repair_model, evaluator, outputs=(0,1), max_depth=None, opt=None):
        if isinstance(precondition, Sampler):
            self.sampler = precondition
        else:
            self.sampler = Sampler(precondition)
        self.original_program = program
        self.repair_model = repair_model
        self.evaluator = evaluator
        self.outputs = outputs
        
        # Maintain an (auxiliary) explicit representation of the tree as a map from binary strings to nodes
        self.tree = {} # Note this will also include non-explored nodes in the worklist
        # The original program forms the root of the tree
        self.root = Node(path=(),solution=repair_model.initial_solution(program))
        self.tree[()] = self.root

        self.max_depth = max_depth # We only consider constraint strings with at most this length (inclusive)
        self.depth = 0 # Bounds the largest constraint string explored (dynamically increases)

        self.opt = opt # Whether to do level-order traversal or Hamming distance heuristic
        # worklist contains (yet-unexplored) children of existing leaves
        if opt is None: # Do an in-order traversal
            valuation = lambda n : len(n.path) # Nodes are sorted by their depth
            threshold = lambda d : d # We will use depth itself as the threshold
        else: # Otherwise it is the threshold ratio
            assert self.max_depth is not None
            self.original_labeling = [self.original_program(*self.sampler.get(i)) for i in range(self.max_depth)]
            # Nodes are sorted by their Hamming distance from the original program
            valuation = lambda n : len([i for i in range(len(n.path)) if n.path[i] != self.original_labeling[i]])
            threshold = lambda d : self.opt * d # We use a fraction of the depth as the threshold
        self.worklist = _HeapQueue(self.depth, valuation, threshold)
        self._add_children(self.root)

    def soln_gen(self):
        while self.worklist.qsize() > 0: # XXX Never terminates if worklist threshold always blocks some nodes

            leaf = self.worklist.get()
            if leaf is None: # We need to expand the depth of the search
                self.depth += 1 # We handle comparing to self.max_depth in _add_children
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
                    leaf.solution.post = self.evaluator.compute_post(leaf.solution.prog)
                    if leaf.solution.post: # Only compute error for correct solutions
                        leaf.solution.error = self.evaluator.compute_error(leaf.solution.prog)
            
            yield leaf

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
