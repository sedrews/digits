from z3 import *
from itertools import product
from digits.parse import parse_fr

# Given phi(hs,xs,y) describing concept class (parameterized by hs; concepts map xs to y)
# return the formula satisfiable iff phi shatters a set of size n
def phi_vc_n(phi,n,hs,xs,y):
    xss = [[Real(str(x)+'_c'+str(i)) for x in xs] for i in range(n)]
    shatter = []
    for bv in product([0,1], repeat=n):
        shatter.append(dich(phi,hs,xs,xss,y,bv))
    vcn = Exists([xss[i][j] for i,j in product(range(n),range(len(xs)))], And(*shatter))
    return vcn

def dich(phi,hs,xs,xss,y,bv):
    assert len(xss) == len(bv)
    subs = []
    for i in range(len(bv)):
        pairs = [(xs[j],xss[i][j]) for j in range(len(xs))] + [(y,RealVal(bv[i]))]
        subs.append(substitute(phi, pairs))
    return Exists(hs, And(*subs))

def calc_vc(phi,hs,xs,y):
    vcn = lambda n : phi_vc_n(phi,n,hs,xs,y)
    i = 1
    while True:
        print("Query for", i, "...")
        s = Solver()
        s.add(vcn(i))
        res = s.check()
        print("  result:", res)
        if res == unsat:
            break
        i += 1
    return i - 1

def fr_vc(filename):
    f = open(filename, 'r')
    c = f.read()
    f.close()

    p = parse_fr(c)
    phi = Tactic('qe').apply(p.D_z3).as_expr()
    hs = p.z3_vars.holes
    xs = p.z3_vars.inputs
    y = p.z3_vars.output

    return calc_vc(phi,hs,xs,y)

if __name__ == '__main__':
    x,y,lx,ux,ly,uy,r = Reals('x y lx ux ly uy r')
    #squares = And(Implies(    And(lx <= x, x <= ux, ly <= y, y <= uy) , r == 1), \
    #              Implies(Not(And(lx <= x, x <= ux, ly <= y, y <= uy)), r == 0))
    squares = And(lx <= x, x <= ux, ly <= y, y <= uy) == (r == 1)
    #print(phi_vc_n(squares,3,[lx,ux,ly,uy],[x,y],r))
    #vcn = lambda n : phi_vc_n(squares,n,[lx,ux,ly,uy],[x,y],r)
    #i = 1
    #while True:
    #    print("Query for", i, "...")
    #    s = Solver()
    #    s.add(vcn(i))
    #    res = s.check()
    #    print("  result:", res)
    #    if res == unsat:
    #        break
    #    i += 1
    calc_vc(squares,[lx,ux,ly,uy],[x,y],r)
