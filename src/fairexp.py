from probexp import ProbPost,_Event,_AExp,_BExp,_NodeType


# minority and hired are lambda functions that take a single input v_map
#   v_map is some dictionary {"<attribute name>" : <attribute value>}
# constant is the 1-epsilon threshhold such that the condition is:
#   ratio > constant
def group_fairness(hired, minority, constant):
    
    m = _Event(exp_type=_NodeType.VAL, predicate=minority)

    nm = _Event(exp_type=_NodeType.NOT)
    nm.children = [_Event(exp_type=_NodeType.VAL, predicate=minority)]

    hm = _Event(exp_type=_NodeType.AND)
    hm.children = [_Event(exp_type=_NodeType.VAL, predicate=hired),
            _Event(exp_type=_NodeType.VAL, predicate=minority)]

    hnm = _Event(exp_type=_NodeType.AND)
    hnm.children = [_Event(exp_type=_NodeType.VAL, predicate=hired),
            _Event(exp_type=_NodeType.NOT)]
    hnm.children[1].children = [_Event(exp_type=_NodeType.VAL, predicate=minority)]

    pm = _AExp(exp_type=_NodeType.PR, children=[m])
    pnm = _AExp(exp_type=_NodeType.PR, children=[nm])
    phm = _AExp(exp_type=_NodeType.PR, children=[hm])
    phnm = _AExp(exp_type=_NodeType.PR, children=[hnm])

    num = _AExp(exp_type=_NodeType.DIV, children=[phm, pm])
    den = _AExp(exp_type=_NodeType.DIV, children=[phnm, pnm])
    frac = _AExp(exp_type=_NodeType.DIV, children=[num, den])

    comp = _BExp(exp_type=_NodeType.LT,
            children=[_AExp(exp_type=_NodeType.VAL, value=constant), frac])

    post = ProbPost()
    post.root = comp
    post.probs = post._gather_probs(post.root)

    return post


if __name__ == "__main__":

    from collections import namedtuple

    Sample = namedtuple('Sample', 'age sex')

    # iopair is of the form (Sample, bool) where the bool is the result
    minority = lambda iopair : iopair[0].sex != "male"
    hired = lambda iopair : iopair[1]

    post = group_fairness(hired, minority, 0.9)

    def eval_post(post, samples):
        counter = {event : 0 for event in post.probs}
        for sample in samples: 
            e_map = post.evaluate_prob_events(sample)
            for event in e_map:
                counter[event] += 1 if e_map[event] else 0
        counter = {event : counter[event] / len(samples) for event in counter}
        return post.evaluate_expression(counter)

    samples = [(Sample(24,   'male'),  True),
               (Sample(24,   'male'),  True),
               (Sample(40,   'male'), False),
               (Sample(26, 'female'),  True),
               (Sample(38, 'female'), False)]
    assert not eval_post(post, samples)

    del samples[0]
    assert eval_post(post, samples)

    print("Passed tests")
