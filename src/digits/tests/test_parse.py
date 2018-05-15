import unittest

import ast
import astor

from digits.parse import *
from z3 import *

from itertools import product

class TestParser(unittest.TestCase):

    def test_SATransformer_as_strings(self):
        code = """def foo(x):
    y = 1
    z = 1
    y = x
    x = x
    if y > 0:
        x = 2
        z = 2
        if y < x:
            z = 3
    else:
        x = 3
        z = 4
    print(x, y, z)
"""
        node = ast.parse(code)
        SATransformer().visit(node)
        new_code = astor.to_source(node)

        expected_code = """def foo(x):
    y_0 = 1
    z_0 = 1
    y_1 = x
    x_0 = x
    if y_1 > 0:
        x_1 = 2
        z_1 = 2
        if y_1 < x_1:
            z_2 = 3
        else:
            z_2 = z_1
    else:
        x_1 = 3
        z_1 = 4
        z_2 = z_1
    print(x_1, y_1, z_2)
"""
        self.assertEqual(expected_code, new_code)

    def test_z3_subsumes_prog(self):
        code = """def D(x, y):
    y = 1
    z = 1
    if x > y:
        z = 2
    else:
        z = x
    return z
"""
        node = ast.parse(code)
        D_exec,hole_defaults,D_z3,z3_vars = process_D_AST(node)
        xs = [-5,-3,0,1,2]
        ys = [-2,1,0,1,2]
        xvar = z3_vars.inputs[0]
        yvar = z3_vars.inputs[1]
        zvar = z3_vars.output
        for x,y in product(xs,ys):
            z = D_exec(x,y)
            s = Solver()
            s.add(D_z3, xvar == x, yvar == y, zvar == z)
            self.assertEqual(s.check(), sat)
            # Can also do some testing that the z3 formula does not contain erroneous solutions
            s = Solver()
            s.add(D_z3, xvar == x, yvar == y, zvar == z + 1)
            self.assertEqual(s.check(), unsat)


if __name__ == '__main__':
    unittest.main()
