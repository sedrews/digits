from collections import namedtuple
import csv


class RunStats:

    def __init__(self, filename):
        f = open(filename, 'r')
        data = list(csv.reader(f))
        f.close()

        self.cmd = data[0][0]
        self.lines = [RunStats._parseline(line) for line in data[1:]]

        self._maxdepth = None
        self._synthstats = None
        self._evalstats = None
        self._beststats = None
        self._threshstats = None

    # Largest depth finished in the search
    @property
    def maxdepth(self):
        if self._maxdepth is None:
            m = -1
            for line in self.lines:
                if not isinstance(line, RunStats.FinishedDepth):
                    continue
                m = line.depth
            self._maxdepth = m
        return self._maxdepth

    # Synthesizer stats as a function of depth/time (as a tuple)
    @property
    def synthstats(self):
        if self._synthstats is None:
            Stats = namedtuple('SynthStats', ['depth', 'time', 'stats'])
            s = []
            for line in self.lines:
                if isinstance(line, RunStats.FinishedDepth):
                    last = line
                elif isinstance(line, RunStats.SynthesizerStats):
                    s.append(Stats(depth=last.depth, time=last.time, stats=line))
            i = 0
            for stat in s:
                assert stat.depth == i
                i += 1
            assert i == self.maxdepth + 1
            self._synthstats = tuple(s)
        return self._synthstats

    # Evaluator stats as a function of depth/time (as a tuple)
    @property
    def evalstats(self):
        if self._evalstats is None:
            Stats = namedtuple('EvalStats', ['depth', 'time', 'stats'])
            s = []
            for line in self.lines:
                if isinstance(line, RunStats.FinishedDepth):
                    last = line
                elif isinstance(line, RunStats.EvaluatorStats):
                    s.append(Stats(depth=last.depth, time=last.time, stats=line))
            i = 0
            for stat in s:
                assert stat.depth == i
                i += 1
            assert i == self.maxdepth + 1
            self._evalstats = tuple(s)
        return self._evalstats

    # Best solution found as a function of depth/time (as a tuple)
    @property
    def beststats(self):
        if self._beststats is None:
            Stats = namedtuple('BestStats', ['depth', 'time', 'stats'])
            s = []
            for line in self.lines:
                if isinstance(line, RunStats.FinishedDepth):
                    last = line
                elif isinstance(line, RunStats.NewBest):
                    s.append(Stats(depth=last.depth, time=line.time, stats=line))
            self._beststats = tuple(s)
        return self._beststats
    
    # Value of the search threshold as a function of depth/time (as a tuple)
    @property
    def threshstats(self):
        if self._threshstats is None:
            Stats = namedtuple('ThreshStats', ['depth', 'time', 'thresh'])
            s = []
            for line in self.lines:
                if isinstance(line, RunStats.FinishedDepth):
                    last = line
                elif isinstance(line, RunStats.UpdatedThresh):
                    s.append(Stats(depth=last.depth, time=line.time, thresh=line.thresh))
            self._threshstats = tuple(s)
        return self._threshstats

    # A bunch of classes for line contents
    InitialOverhead = namedtuple('InitialOverhead', ['time'])
    EnteredGenerator = namedtuple('EnteredGenerator', ['time'])
    FinishedDepth = namedtuple('FinishedDepth', ['time', 'depth'])
    SynthesizerStats = namedtuple('SynthesizerStats', ['time', 'calls', 'smt', 'sat', 'unsat', 'pruned', 'make_time', 'sanity_time', 'avg_num_constraints'])
    EvaluatorStats = namedtuple('EvaluatorStats', ['time', 'fast_post_calls', 'fast_post_time', 'fast_error_calls', 'fast_error_time', 'slow_post_calls', 'slow_post_time', 'slow_error_calls', 'slow_error_time'])
    NewBest = namedtuple('NewBest', ['time', 'error', 'path_length', 'valuation'])
    UpdatedThresh = namedtuple('UpdatedThresh', ['time', 'thresh'])
    TimedOut = namedtuple('TimedOut', ['time'])
    ExhaustedGenerator = namedtuple('ExhaustedGenerator', ['time'])
    Best = namedtuple('Best', ['time', 'error', 'holes'])

    @staticmethod
    def _parseline(line):
        if line[0] == 'initial overhead': # This is the only weird case
            return RunStats.InitialOverhead(time=float(line[1]))
        time = float(line[0])
        idstr = line[1]
        if idstr == 'entered generator':
            return RunStats.EnteredGenerator(time=time)
        elif idstr == 'finished depth':
            return RunStats.FinishedDepth(time=time, depth=int(line[2]))
        elif idstr == 'synthesizer stats':
            return RunStats.SynthesizerStats(time=time, calls=int(line[3]), smt=int(line[5]), sat=int(line[7]), unsat=int(line[9]), pruned=int(line[11]), make_time=float(line[13]), sanity_time=float(line[15]), avg_num_constraints=float(line[17]))
        elif idstr == 'evaluator stats':
            return RunStats.EvaluatorStats(time=time, fast_post_calls=int(line[3]), fast_post_time=float(line[5]), fast_error_calls=int(line[7]), fast_error_time=float(line[9]), slow_post_calls=int(line[11]), slow_post_time=float(line[13]), slow_error_calls=int(line[15]), slow_error_time=float(line[17]))
        elif idstr == 'new best':
            return RunStats.NewBest(time=time, error=float(line[2]), path_length=int(line[4]), valuation=int(line[6]))
        elif idstr == 'updated thresh':
            return RunStats.UpdatedThresh(time=time, thresh=float(line[2]))
        elif idstr == 'timed out':
            return RunStats.TimedOut(time=time)
        elif idstr == 'exhausted generator':
            return RunStats.ExhaustedGenerator(time=time)
        elif idstr == 'best':
            return RunStats.Best(time=time, error=float(line[3]), holes=tuple(float(h) for h in line[5:]))
        else:
            assert False, "bad line message: " + str(idstr)
