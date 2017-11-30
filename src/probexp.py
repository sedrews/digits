from enum import Enum
from functools import reduce


# A class to represent probabilistic postconditions
#
# Post   := BExp
# BExp   := True | False
#         | BExp and BExp | BExp or BExp | not BExp
#         | AExp < AExp
# AExp   := c
#           (for some constant c)
#         | Pr[Event]
#         | AExp + AExp | AExp - AExp | AExp * AExp | AExp / AExp
# Event  := p
#           (for some predicate p over input variables and output)
#         | Event and Event | Event or Event | not Event
class ProbPost:
    
    def __init__(self):
        self.root = None # TODO eventually of type _BExp
        self.probs = [] # A list of the probabilty expressions in post (TODO build it)
                        # i.e., each element is the _Event child of some _AExp w/ type PR
    
    # Map a Pr->R assignment to True/False
    def evaluate_expression(self, pr_map):
        return self.root.evaluate_expression(pr_map)

    # Map variable values to an event assignment
    def evaluate_prob_events(self, v_map):
        return {e : e.evaluate_prob_event(v_map) for e in self.probs}

    def _gather_probs(self, node):
        if node.exp_type == _NodeType.PR:
            assert(len(node.children) == 1)
            return [node.children[0]]
        try:
            return reduce(lambda x,y : x + y, [self._gather_probs(child) for child in node.children])
        except AttributeError:
            return []


# The internal structures for the expression


class _NodeType(Enum):
    AND = 1
    OR = 2
    NOT = 3
    LT = 4
    TRUE = 5
    FALSE = 6
    VAL = 7
    PR = 8
    ADD = 9
    SUB = 10
    MUL = 11
    DIV = 12


class _Node(object):

    __slots__ = 'children','exp_type'

    def __init__(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self, key, value)


class _Event(_Node):

    __slots__ = 'predicate'

    def evaluate_prob_event(self, v_map):
        if self.exp_type == _NodeType.VAL:
            return self.predicate(v_map)
        elif self.exp_type == _NodeType.AND:
            assert(len(self.children) > 1)
            return reduce(lambda x,y : x and y, [child.evaluate_prob_event(v_map) for child in self.children])
        elif self.exp_type == _NodeType.OR:
            assert(len(self.children) > 1)
            return reduce(lambda x,y : x or y, [child.evaluate_prob_event(v_map) for child in self.children])
        elif self.exp_type == _NodeType.NOT:
            assert(len(self.children) == 1)
            return not self.children[0].evaluate_prob_event(v_map)
        else:
            assert(False)


class _AExp(_Node):

    __slots__ = 'value'
    
    def evaluate_expression(self, pr_map):
        if self.exp_type == _NodeType.VAL:
            return self.value
        elif self.exp_type == _NodeType.PR:
            assert(len(self.children) == 1 and self.children[0].__class__ == _Event)
            return pr_map[self.children[0]]
        elif self.exp_type == _NodeType.ADD:
            assert(len(self.children) > 1)
            return reduce(lambda x,y : x + y, [child.evaluate_expression(pr_map) for child in self.children])
        elif self.exp_type == _NodeType.SUB:
            assert(len(self.children) == 2)
            return self.children[0].evaluate_expression(pr_map) - self.children[1].evaluate_expression(pr_map)
        elif self.exp_type == _NodeType.MUL:
            assert(len(self.children) > 1)
            return reduce(lambda x,y : x * y, [child.evaluate_expression(pr_map) for child in self.children])
        elif self.exp_type == _NodeType.DIV:
            assert(len(self.children) == 2)
            return self.children[0].evaluate_expression(pr_map) / self.children[1].evaluate_expression(pr_map)
        else:
            assert(False)


class _BExp(_Node):
    
    def evaluate_expression(self, pr_map):
        if self.exp_type == _NodeType.AND:
            assert(len(self.children) > 1)
            return reduce(lambda x,y : x and y, [child.evaluate_expression(pr_map) for child in self.children])
        elif self.exp_type == _NodeType.OR:
            assert(len(self.children) > 1)
            return reduce(lambda x,y : x or y, [child.evaluate_expression(pr_map) for child in self.children])
        elif self.exp_type == _NodeType.NOT:
            assert(len(self.children) == 1)
            return not self.children[0].evaluate_expression(pr_map)
        elif self.exp_type == _NodeType.LT:
            assert(len(self.children) == 2)
            return self.children[0].evaluate_expression(pr_map) < self.children[1].evaluate_expression(pr_map)
        elif self.exp_type == _NodeType.TRUE:
            return True
        elif self.exp_type == _NodeType.FALSE:
            return False
        else:
            assert(False)
