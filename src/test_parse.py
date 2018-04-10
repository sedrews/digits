import unittest

import ast
import astor
from parse import SATransformer

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

if __name__ == '__main__':
    unittest.main()
