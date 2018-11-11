# A hastily-written cvc4 interface
# XXX this is all a hack

from functools import reduce
import subprocess

cvc4_path = None
def set_path(path):
    global cvc4_path
    cvc4_path = path

def cvc4_query_from_z3(s, model_vars, names):
    smt2 = s.to_smt2()
    in_lines = [line for line in [ell.strip() for ell in smt2.split('\n')] if len(line) > 0]
    modify_lines(in_lines, model_vars, names)
    in_str = reduce(lambda x,y: x + "\n" + y, in_lines)
    out_str = call_cvc4(in_str)
    out_lines = [line for line in [ell.strip() for ell in out_str.split('\n')] if len(line) > 0]
    model = try_model_extract(out_lines, model_vars)
    core = try_core_extract(out_lines, names)
    return model, core

def call_cvc4(in_str):
    assert cvc4_path is not None
    # XXX in cvc4-1.6, '--output-lang z3str' doesn't seem to change anything
    res = subprocess.run([cvc4_path, "--lang", "smt2"],
                         input=in_str, encoding='ascii',
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert len(res.stderr) == 0
    return res.stdout

def try_model_extract(lines, model_vars):
    try:
        assert lines[0] == "sat"
        assignments = lines[1:len(model_vars)+1]
        parsed = [parse_assignment(a) for a in assignments]
        parsed_map = {name : (num, den) for name,num,den in parsed}
        for v in model_vars:
            assert v in parsed_map
        return parsed_map
    except AssertionError:
        pass

def parse_assignment(line):
    # Expected, e.g., "((x (/ (- 1) 1)))"
    ast = match_parentheses(line)
    assert len(ast.children) == 1
    temp = ast.children[0]
    assert len(temp.children) == 1
    temp = temp.children[0]
    assert len(temp.children) == 2
    v = temp.children[0]
    expr = temp.children[1]
    assert v.children is None
    assert len(expr.children) == 3
    div,n,d = tuple(expr.children)
    assert div.children is None and div.value == '/'
    def handle(t):
        if t.children is None:
            return eval(t.value)
        else:
            for child in t.children:
                assert child.children is None
            return eval(reduce(lambda x,y : x + " " + y, [child.value for child in t.children], ""))
    name = v.value
    numerator = handle(n)
    denominator = handle(d)
    return (name, numerator, denominator)


# For matching parentheses
class Node:
    __slots__ = ['parent', 'children', 'value']
    def __init__(self, **kwargs):
        for key in Node.__slots__:
            setattr(self, key, kwargs.get(key))

def match_parentheses(string):
    tokens = []
    while len(string) > 0:
        c = string[0]
        if c == ' ':
            string = string[1:]
        elif c == '(' or c == ')':
            tokens.append(c)
            string = string[1:]
        else:
            i = 1
            while i < len(string) and string[i] not in {' ', '(', ')'}:
                i += 1
            tokens.append(string[:i])
            string = string[i:]
    root = Node(children=[])
    current = root
    for token in tokens:
        if token == '(':
            current.children.append(Node(parent=current,children=[]))
            current = current.children[-1]
        elif token == ')':
            current = current.parent
        else:
            current.children.append(Node(parent=current,value=token))
    assert current == root
    return root

def try_core_extract(lines, names):
    try:
        assert lines[0] == "unsat"
        assert lines[-1] == ")"
        i = len(lines) - 2
        core = []
        while lines[i] != "(":
            assert lines[i][:5] == "name_"
            core.append(lines[i][5:])
            i -= 1
        return core
    except AssertionError:
        pass

def modify_lines(lines, model_vars, bools):
    assert lines[0] == "; benchmark generated from python API"
    assert lines[1] == "(set-info :status unknown)"
    lines.insert(2, "(set-logic LRA)")
    lines.insert(3, "(set-option :produce-models true)")
    lines.insert(4, "(set-option :produce-unsat-cores true)")

    assert lines[-1] == "(check-sat)"
    # z3's Solver.to_smt2() is odd wrt unsat cores, so we do it manually
    for p in bools:
        lines.insert(-1, "(assert (! " + p + " :named name_" + p + "))")
    for v in model_vars:
        lines.append("(get-value (" + v + "))")
    lines.append("(get-unsat-core)")
