from probexp import ProbPost,_Event,_AExp,_BExp,_NodeType


# minority and hired are lambda functions that take a single input v_map
#   v_map is some dictionary {"<attribute name>" : <attribute value>}
# constant is the 1-epsilon threshhold such that the condition is:
#   ratio > constant
def group_fairness(minority, hired, constant):
    
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

    minority = lambda v_map : v_map["sex"] == "female"
    hired = lambda v_map : v_map["output"] > 0

    post = group_fairness(minority, hired, 0.9)

    samples = [{"sex" :   "male", "output" :  True},
               {"sex" :   "male", "output" :  True},
               {"sex" :   "male", "output" : False},
               {"sex" : "female", "output" :  True},
               {"sex" : "female", "output" : False}]
    ec = {v : 0 for v in post.probs}
    for sample in samples: 
        e_map = post.evaluate_prob_events(sample)
        for e in e_map:
            ec[e] += 1 if e_map[e] else 0
    ec = {e : ec[e] / len(samples) for e in ec}
    print(post.evaluate_expression(ec))
