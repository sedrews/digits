import ast
import codegen
import astor
from z3 import *
from collections import namedtuple
from probs import prob_dict


def parse_fr(code_string):
    node = ast.parse(code_string)

    pre,D,post = separate_FR_AST(node)
    pre_exec = process_pre_AST(pre)
    D_exec,hole_defaults,D_z3,z3_vars = process_D_AST(D)
    post_exec = process_post_AST(post)

    ParsedFR = namedtuple('ParsedFR', ['pre_exec', 'D_exec', 'hole_defaults', 'D_z3', 'z3_vars', 'post_exec'])

    return ParsedFR(pre_exec, D_exec, hole_defaults, D_z3, z3_vars, post_exec)


# helpers
def name(node):
    return ast.dump(node).split("(")[0]
# HACK:
def isTrue(node):
    return "True" in ast.dump(node)
# HACK:
def isFalse(node):
    return "False" in ast.dump(node)
def isCall(node):
    return name(node) == 'Call'
def isAssign(node):
    return name(node) == 'Assign'
def isIf(node):
    return name(node) == 'If'
def isExpr(node):
    return name(node) == 'Expr'
def isReturn(node):
    return name(node) == 'Return'
def isBinOp(node):
    return name(node) == 'BinOp'
def isBoolOp(node):
    return name(node) == 'BoolOp'
def isUnaryOp(node):
    return name(node) == 'UnaryOp'
def isCompareOp(node):
    return name(node) == 'Compare'
def isAdd(node):
    return name(node) == 'Add'
def isPow(node):
    return name(node) == 'Pow'
def isAnd(node):
    return name(node) == 'And'
def isOr(node):
    return name(node) == 'Or'
def isUAdd(node):
    return name(node) == 'UAdd'
def isUSub(node):
    return name(node) == 'USub'
def isMult(node):
    return name(node) == 'Mult'
def isDiv(node):
    return name(node) == 'Div'
def isSub(node):
    return name(node) == 'Sub'
def isNot(node):
    return name(node) == 'Not'
def isEq(node):
    return name(node) == 'Eq'
def isNotEq(node):
    return name(node) == 'NotEq'
def isLt(node):
    return name(node) == 'Lt'
def isLtE(node):
    return name(node) == 'LtE'
def isGt(node):
    return name(node) == 'Gt'
def isGtE(node):
    return name(node) == 'GtE'
def isNum(node):
    return name(node) == 'Num'
def isName(node):
    return name(node) == 'Name'

def evalAST(node):
    node = codegen.to_source(node)
    return eval(node)

def makeBin(op, l, r):
    if isAdd(op): return l + r
    elif isSub(op): return l - r
    elif isMult(op): return l * r
    elif isDiv(op): return l / r
    elif isPow(op): return l**r
    else: assert False, "Weird binary op"
def makeUnary(op, e):
    if isUSub(op): return -e
    elif isUAdd(op): return +e
    elif isNot(op): return Not(e)
    else: assert False, "Weird unary op"
def makeBool(op, *args):
    if isAnd(op): return And(*args)
    elif isOr(op): return Or(*args)
    else: assert False, "Weird bool op"
def makeCompare(op, l, r):
    if isGt(op): return l > r
    elif isGtE(op): return l >= r
    elif isLt(op): return l < r
    elif isLtE(op): return l <= r
    elif isEq(op): return l == r
    elif isNotEq(op): return l != r
    else: assert False, "Weird compare"


# Convert the AST to (not-static) single-assignment form
# This facilitates a direct conversion to logical formulae
class SATransformer(ast.NodeTransformer): #XXX probably does not support x += 1 etc

    # The module should be a single function (we assert so, in fact)
    def visit_Module(self, node):
        self.live_index = {} 
        
        assert len(node.body) == 1
        func = node.body[0]
        self.live_inputs = [a.arg for a in func.args.args]
        for name in self.live_inputs:
            self.live_index[name] = -1 # For identifiers as arguments, use base name before name_0
        func.body = self.doSeq(func.body)

        return node

    def doSeq(self, seq):
        return [self.visit(s) for s in seq]

    def visit_If(self, node):

        self.visit(node.test)

        before = self.live_index.copy()
        node.body = self.doSeq(node.body)
        after_if = self.live_index.copy()

        self.live_index = before.copy()
        node.orelse = self.doSeq(node.orelse)
        after_else = self.live_index.copy()
        
        for name in self.live_index:
            # Updated indexes after the join
            self.live_index[name] = max(after_if[name], after_else[name])
            # Add the Phi functions
            if after_if[name] < self.live_index[name]:
                # Add to the end of the if-branch: name_{live} = name_{after_if}
                node.body.append(ast.Assign(targets=[ast.Name(id=self.app_ind(name,self.live_index))],
                                            value=ast.Name(id=self.app_ind(name,after_if))))
            if after_else[name] < self.live_index[name]:
                # Add to the end of the else-branch: name_{live} = name_{after_else}
                node.orelse.append(ast.Assign(targets=[ast.Name(id=self.app_ind(name,self.live_index))],
                                              value=ast.Name(id=self.app_ind(name,after_else))))

        return node

    def visit_Assign(self, node):
        assert len(node.targets) == 1
        value = self.visit(node.value)
        t = node.targets[0].id
        if t in self.live_index:
            self.live_index[t] += 1
        else:
            self.live_index[t] = 0
        target = self.visit(node.targets[0])
        return ast.Assign(targets=[target], value=value)

    def visit_Call(self, node):
        node.args = [self.visit(arg) for arg in node.args]
        return node

    def visit_Name(self, node):
        return ast.Name(id=self.app_ind(node.id,self.live_index))

    def app_ind(self, name, index_dict):
        ind = index_dict[name]
        if ind == -1: # This is convention for the case in which arguments to functions are left alone
            return name
        else:
            return name + "_" + str(ind)


# Essentially syntactic conversion of an SA-form Python program
# to a z3 formula
class Z3Encoder(ast.NodeVisitor):

    def __init__(self, inputs=None, holes=None):
        self.phi = None
        self.holes = holes if holes is not None else []
        self.inputs = inputs if inputs is not None else []
        self.retvar = None
        self.othervars = []

    def generic_visit(self, node):
        assert False, "generic visit on node " + ast.dump(node)

    def visit_Module(self, node):
        assert len(node.body) == 1
        f = node.body[0]
        assert f.name == 'D'

        self.phi = self.doSeq(f.body)
        self.phi = simplify(self.phi)

        if self.retvar in self.othervars:
            self.othervars.remove(self.retvar)

    def doSeq(self, seq):
        trans = [self.visit(s) for s in seq]
        return simplify(And(*trans))

    def visit_Assign(self, node):
        assert len(node.targets) == 1, "No unpacked tuple assignments allowed"

        lhs = node.targets[0].id
        rhs = node.value

        zrhs = exprToZ3(rhs)
        zlhs = Real(lhs)

        if lhs in self.holes:
            pass
        else:
            if lhs not in self.othervars:
                self.othervars.append(lhs)

        return zlhs == zrhs

    def visit_If(self, node):
        zcond = exprToZ3(node.test)
        zthen = self.doSeq(node.body)
        zelse = self.doSeq(node.orelse)
        return And(Implies(zcond, zthen), Implies(Not(zcond), zelse))

    # We handle expressions in If and Assign statements in their visit_'s,
    # so these are always calls
    def visit_Expr(self, node):
        assert isinstance(node.value, ast.Call), "Unexpected expression"
        return True

    # These are always the event annotations
    def visit_Call(self, node):
        fn = node.func.id
        assert len(node.args) == 1

        if fn == 'event':
            pass
        else:
            assert False, "Unrecognizable function call"

        return True

    def visit_Return(self, node):
        assert isinstance(node.value, ast.Name), "Must return an identifier (bind any expression in a previous line)"
        self.retvar = node.value.id
        return True


# e is an ast expr node
def exprToZ3(e):
    if isBinOp(e):
        zlhs = exprToZ3(e.left)
        zrhs = exprToZ3(e.right)
        return makeBin(e.op, zlhs, zrhs)
    elif isUnaryOp(e):
        zexp = exprToZ3(e.operand)
        return makeUnary(e.op, zexp)
    elif isBoolOp(e):
        zexprs = [exprToZ3(v) for v in e.values]
        return makeBool(e.op, zexprs)
    elif isCompareOp(e):
        assert len(e.ops) == 1
        zlhs = exprToZ3(e.left)
        zrhs = exprToZ3(e.comparators[0])
        return makeCompare(e.ops[0], zlhs, zrhs)
    elif isNum(e):
        return evalAST(e)
    elif isName(e):
        if isTrue(e):
            return BoolVal(True)
        elif isFalse(e):
            return BoolVal(False)
        else:
            return Real(e.id)
    else:
        assert False, "Weird expression" + ast.dump(e)

def separate_FR_AST(node):
    # Return pre, D, and post AST objects
    assert isinstance(node, ast.Module)
    assert len(node.body) == 3
    pre = node.body[0]
    D = node.body[1]
    post = node.body[2]
    assert pre.name == 'pre' and D.name == 'D' and post.name == 'post'
    return ast.Module(body=[pre]), ast.Module(body=[D]), ast.Module(body=[post])

def process_pre_AST(node):
    # Call eval, ensuring our support prob dists are in globals namespace
    # Return the executable
    c = compile(node, '<string>', mode='exec')
    m = {}
    eval(c, prob_dict, m)
    pre = m['pre']
    def wrapped():
        inputs = pre()
        return tuple(Fraction(val) for val in inputs)
    return wrapped

def process_D_AST(node):
    # Replace the AST Hole() calls with fresh variables
    # For executable:
    #   change args from (*inputs) to (*inputs, *holevals, eventfunc)
    #   evaluate this function
    #   wrap (statically) in a function : (*inputs, *holevals) -> res,event_map
    # For z3:
    #   single-assignment transformation
    #   syntactic z3 conversion
    # Return the following:
    #   the z3 formula and the input, ouput, hole, and intermediary variables
    #   the executable and the default hole values

    inputs = [a.arg for a in node.body[0].args.args]

    h = HoleCallTransformer()
    h.visit(node)

    class ArgTransformer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            assert node.name == 'D'
            node.args.args = [ast.arg(arg=x, annotation=None) for x in ["event"] + h.holes] + node.args.args
            return node
    ArgTransformer().visit(node)

    ast.fix_missing_locations(node)

    #print("process_D_AST debug:")
    #print(astor.to_source(node))

    c = compile(node, '<string>', mode='exec')
    m = {}
    eval(c, None, m)
    D = m['D']
    def wrapped(*xs):
        d = {}
        def event(event_name, bool_val):
            # It should be an invariant that each event is encountered
            # exactly once during the program execution
            # (otherwise semantics don't really make sense).
            #TODO make sure that if the event isn't encountered that something appropriate happens
            assert event_name not in d
            d[event_name] = bool_val
        ret = D(event, *xs)
        return ret, d
    #XXX if this function is called a second time, will the previously
    #    returned wrapped function (incorrectly) use the new m['D']?
    #    Similar concern in process_pre_AST

    z = Z3Encoder(inputs, h.holes)
    z.visit(SATransformer().visit(node))
    
    ZT = namedtuple("Z3_vars", ['inputs', 'holes', 'output', 'intermediary'])
    zt = ZT([Real(i) for i in z.inputs], [Real(i) for i in z.holes], Real(z.retvar), [Real(i) for i in z.othervars])
    phi = z.phi if len(zt.intermediary) == 0 else Exists(zt.intermediary, z.phi)
    return wrapped,(h.holes, h.hole_map),phi,zt


class HoleCallTransformer(ast.NodeTransformer):

    def __init__(self):
        super().__init__()
        self.hole_map = {}
        self.holes = []

    def next_name(self):
        return "x__hole_" + str(len(self.holes))

    def visit_Module(self, node):
        assert len(node.body) == 1
        func = node.body[0]
        func.body = self.doSeq(func.body)

        return node

    def doSeq(self, seq):
        return [self.visit(s) for s in seq]

    def visit_Call(self, node):
        if node.func.id == "Hole":
            assert len(node.args) == 1
            arg = evalAST(node.args[0])
            name = self.next_name()
            self.holes.append(name)
            self.hole_map[name] = arg
            return ast.Name(id=name, ctx=ast.Load())
        else:
            return node


def process_post_AST(node):
    # eval, potentially including some Pr() : joint_counts_dict -> prob in the global namespace? TODO
    # Return that function
    c = compile(node, '<string>', mode='exec')
    m = {}
    eval(c, None, m)
    return m['post']

if __name__=="__main__":

    node = ast.parse('''
def pre():
    x = step([(-1, 1, 1)])
    y = step([(-1, 1, 1)])
    return x, y

def D(x, y):
    if Hole(-.5) < x and x < Hole(.7) and Hole(0) < y and y < Hole(1):
        ret = 1
    else:
        ret = 0
    event("neg_x", x < 0)
    event("in_box", ret == 1)
    return ret

def post(Pr):
    num = Pr({"in_box" : True, "neg_x" : True}) / Pr({"neg_x" : True})
    den = Pr({"in_box" : True, "neg_x" : False}) / Pr({"neg_x" : False})
    ratio = num / den
    return ratio > 0.95
    ''')

    pre,D,post = separate_FR_AST(node)
    pre_exec = process_pre_AST(pre)
    for i in range(100):
        print(pre_exec())
    D_exec,hole_defaults,D_z3,z3_vars = process_D_AST(D)
    print("D_z3?")
    print(D_z3)
    print("z3_vars?", z3_vars)
    print("hole_defaults", hole_defaults)
    print("some D_exec trials")
    for i in range(10):
        inputs = pre_exec()
        outputs = D_exec(*[hole_defaults[1][h] for h in hole_defaults[0]], *inputs)
        print("mapped", inputs, "to", outputs)
    post_exec = process_post_AST(post)

    # TODO error detection:
    # Prob sampling should only be in pre
    # Ensure #args in D matches #returns in pre
    # Ensure the variable names introduced in D for holes don't have collisions
    # event() should only be in D
    # Ensure all events are event(string literal, bool exp)
    # Ensure post uses only defined events

    # TODO make an internal transformation of all program identifiers
    # to some standard generic template -- this way, we can
    # introduce variable names without worrying about conflict.
