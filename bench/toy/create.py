from functools import reduce
from itertools import product

def build(dims, epsilon):
    return build_pre(dims) + "\n" + build_body(dims, epsilon) + "\n" + build_post(epsilon)

def varlist(dims):
    assert dims > 0
    ret = "x0"
    for i in range(1,dims):
        ret += ", x" + str(i)
    return ret

def build_pre(dims):
    assert dims > 0
    code_str = "def pre():\n"
    for i in range(dims):
        code_str += "    x" + str(i) + " = step([(-1,1,1)])\n"
    code_str += "    return " + varlist(dims) + "\n"
    return code_str

def build_body(dims, epsilon):
    assert dims > 0
    code_str = "def D(" + varlist(dims) + "):\n"
    code_str += "    event(\"pos\", x0 >= 0)\n"
    code_str += "    if Hole(0) <= x0 and x0 <= Hole(" + str(epsilon) + ")"
    for i in range(1,dims):
        code_str += " and Hole(-1) <= x" + str(i) + " and x" + str(i) + " <= Hole(1)"
    code_str += ":\n"
    code_str += "        ret = 1\n"
    code_str += "    else:\n"
    code_str += "        ret = 0\n"
    code_str += "    event(\"accept\", ret == 1)\n"
    code_str += "    return ret\n"
    return code_str

def build_post(epsilon):
    code_str = "def post(Pr):\n"
    code_str += "    neg = Pr({\"accept\" : True, \"pos\" : False})\n"
    code_str += "    pos = Pr({\"accept\" : True, \"pos\" : True})\n"
    code_str += "    accept = Pr({\"accept\" : True}) >= " + str(epsilon / 2) + "\n"
    code_str += "    return neg >= pos and accept\n"
    return code_str

if __name__ == '__main__':
    dimensions = [1,2,3] # VC dimension is dim*2
    bounds = [.1,.2,.4,.8] # Optimal solution is at bound/2
    for d,b in product(dimensions, bounds):
        fname = "box_d" + str(d) + "_b" + str(b).replace(".", "") + ".fr"
        f = open("fr/" + fname, 'w')
        f.write(build(d,b))
        f.close()
    #print(build(3,.5))
