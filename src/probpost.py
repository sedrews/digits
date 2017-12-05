from functools import reduce

# A class to represent (and evaluate) probabilistic postconditions
class ProbPost:

    def __init__(self):
        # A dictionary from str names to functions
        # Example: {"hired" : lambda iopair : iopair[1],
        #           "minority" : lambda iopair : iopair[0].x < 0}
        self.preds = None

        # A list of Event objects:
        # each contains a hashable (partial) map from predicate strings to True/False
        # Example: [Event({"hired" : True, "minority" : True}),
        #           Event({"minority" : True}),
        #           Event({"hired" : True, "minority" : False}),
        #           Event({"minority" : False})]
        self.events = None

        # A function that takes as input a dictionary
        # from self.events to a probability assignment
        # Example:
        # def post(pr_map):
        #     num = pr_map[{"hire" : True, "minority" : True}]
        #         / pr_map({"minority" : True})
        #     den = pr_map[{"hire" : True, "minority" : False}]
        #         / pr_map({"minority" : False})
        #     ratio = num / den
        #     return ratio > 0.95
        self.func = None

    # pr_map is a dictionary
    #   from an event (i.e. dictionary : str -> bool)
    #   to a probability value for that event
    def check(self, pr_map):
        try:
            val = self.func(pr_map)
        except ZeroDivisionError:
            val = None
        return val

    # values is the input to each of the self.preds functions
    def evaluate_events(self, values):
        pred_values = {p : self.preds[p](values) for p in self.preds}
        return {e : reduce(lambda x,y : x and y, [pred_values[p] == e.assignment[p] for p in e.assignment]) for e in self.events}

# Don't mutate these
class Event:

    def __init__(self, assignment):
        self._hash = None
        self.assignment = assignment

    def __eq__(self, other):
        assert isinstance(other, Event)
        return self.assignment == other.assignment
    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if self._hash is None:
            val = 0
            for pair in self.assignment.items():
                val ^= hash(pair)
            self._hash = val
        return self._hash
