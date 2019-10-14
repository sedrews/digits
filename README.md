# DIGITS (DIstribution-Guided InducTive Synthesis)
The prototype implmentation of the probabilistic synthesis tool as seen in ["Efficient Synthesis with Probabilistic Constraints" (CAV19)](https://doi.org/10.1007/978-3-030-25540-4_15) and ["Repairing Decision-Making Programs Under Uncertainty" (CAV17)](https://doi.org/10.1007/978-3-319-63387-9_9).

## Installation
This prototype tool runs in python (python3.6 or newer) and has a number of dependencies:
* We use [numpy/scipy](https://www.scipy.org/install.html) to sample from probability distributions and evaluate their CDFs.
* We use the [Z3](https://github.com/Z3Prover/z3) python bindings for individual synthesis queries.  While it is necessary to have Z3 installed, the tool also supports using [CVC4](https://cvc4.github.io/) for the SMT backend (use the --cvc4 flag to point to the binary).
* We use the [gmpy2](https://gmpy2.readthedocs.io/) high-precision arithmetic library for python.
* We use the [astor](https://pypi.org/project/astor/) library as part of parsing our .fr input files.

Many of these dependencies can be installed using pip:
```
pip install --user numpy scipy gmpy2 astor
```
though note that `pip install gmpy2` will fail without several backend development libraries:
`apt-get install libgmp-dev libmpfr-dev libmpc-dev`.

## Getting Started
The file [src/run.py](src/run.py) connects the backend [digits package](src/digits) to some argument parsing.
As an example, [src/rect_test.fr](src/rect_test.fr) contains the specification for a synthesis/repair problem---namely,
the program sketch `D(x, y)` contains four "holes" that our synthesizer should instantiate.
To invoke the DIGITS algorithm on this problem (using 5 samples from the precondition),
we would (first `cd src` and then) run the following:
```
python run.py -f rect_test.fr -d 5
```
After a few seconds and many lines of output, DIGITS should output a final line looking something like
```
best error 0.39... holes [-0.37..., 0.26..., -0.92..., 0.31...]
```
denoting that the best solution it found, whose quantitative objective function had value 0.39,
corresponds to instantiating the holes in `D(x, y)` with the given values in the bracketed list
(in the order in which they were parsed).
