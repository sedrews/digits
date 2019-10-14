This directory contains the benchmarks from the CAV19 paper, which fall into three categories:
* [orig/](orig/): Comparing the improved synthesis technique (CAV19) against the original digits algorithm (CAV17)
on the fairness repair benchmarks.
* [toy/](toy/): Evaluating the parameters of the synthesis technique on a class of toy examples with known optimal solutions.
* [therm/](therm/): Attempting to synthesize a sketched thermostat controller (challenging due to a deep loop unrolling).

To recreate the experiments from the the paper, do each of
```
cd orig
./run.sh listf.txt listp.txt res
cd ../toy
./run.sh listf.txt listp.txt res
cd ../therm
./run.sh res [optional, but recommended, path to CVC4 binary]
```

Though note that without changing the parameters (options in the listf.txt and listp.txt files),
this full suite of benchmarks will take around a day (~21 hours for `orig/` alone, due to the allowed time for each search).

Once they are created, summaries of the runs can be printed to the screen using
```
python main.py
```

Note that generate some of the graphs was partially automated with `graphs.py`,
but changes to the output format have broken the script.
