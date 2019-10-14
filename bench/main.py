from collections import namedtuple
from functools import reduce
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as be_pdf

from dataparser import RunStats


def therm():
    prefix = "therm/res/"
    unrollings = [40,20,10,5]
    bounds = [8,4,2]
    fname = lambda u,b: "therm_u" + str(u) + "_b" + str(b) + ".jsonl"
    stats = {(u,b) : RunStats(prefix + fname(u,b)) for u,b in product(unrollings, bounds)}
    ThermData = namedtuple('ThermData', ['unrolling', 'bound', 'depth', 'error'])
    return [ThermData(u, b, stats[(u,b)].depth, stats[(u,b)].best) for u,b in product(unrollings, bounds)]

def orig():
    prefix = "orig/res/"
    bench = ["dt4", "dt16", "dt44",
             "svm3-1", "svm3-2", "svm3-3", "svm3-4",
             "svm4-1", "svm4-2", "svm4-3", "svm4-4", "svm4-5",
             "svm5-1", "svm5-2", "svm5-3", "svm5-4", "svm5-5", "svm5-6"]
    options = ["o1", "a1"]
    fname = lambda b,o : b + "_s0_" + o + ".jsonl"
    stats = {(b,o) : RunStats(prefix + fname(b,o)) for b,o in product(bench, options)}
    OrigData = namedtuple('OrigData', ['benchmark', 'original_depth', 'original_error',
                                                    'adaptive_depth', 'adaptive_error'])
    return [OrigData(b, stats[(b,"o1")].depth, stats[(b,"o1")].best,
                        stats[(b,"a1")].depth, stats[(b,"a1")].best) for b in bench]

def toy():
    prefix = "toy/res/"
    dimension = [1,2,3]
    initial = ["02", "04", "08"]
    options = ["o1", "o03", "o015", "o007", "a1"]
    fname = lambda d,i,o : "box_d" + str(d) + "_b" + i + "_s0_" + o + ".jsonl"
    stats = {(d,i,o) : RunStats(prefix + fname(d,i,o)) for d,i,o in product(dimension, initial, options)}
    ToyData = namedtuple('ToyData', ['benchmark', 'depth', 'error'])
    return [ToyData(fname(d,i,o), stats[(d,i,o)].depth, stats[(d,i,o)].best) for d,i,o in product(dimension, initial, options)]

def foo():
    prefix = "toy/res/"
    dimension = [1,2,3]
    initial = ["01", "02", "04", "08"]
    options = ["o1", "o05", "o03", "o015", "o007", "a1"]
    fname = lambda d,i,o : "box_d" + str(d) + "_b" + i + "_s0_" + o + ".jsonl"
    index = 0
    for d,i in product(dimension, initial):
        runs = {}
        for option in options:
            stats = RunStats(prefix + fname(d,i,option))
            error_vs_time = [(t,e) for t,d,e in stats.temporal_error_data]
            depth_vs_time = stats.temporal_depth_data
            depth_vs_time += [(120, depth_vs_time[-1].depth)] # Add on a point for the depth it was working on
            runs[option] = (error_vs_time, depth_vs_time)
        error_depth_time_plot(options, runs, index)
        plt.suptitle(fname(d,i,"*"))
        index += 1
    pdf = be_pdf.PdfPages("foo.pdf")
    for n in range(index):
        pdf.savefig(plt.figure(n))
    pdf.close()

def error_depth_time_plot(options, runs, num):
    fig, ax1 = plt.subplots(num=num)
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('best error')
    ax1.set_xlim([0,120])
    ax1.set_ylim([0,.5])
    ax2 = ax1.twinx()
    ax2.set_ylabel('depth')
    for option in options:
        tes,tds = runs[option]
        ax1.plot([time for time,error in tes], [error for time,error in tes], 'o-')
        ax2.step([time for time,depth in tds], [depth for time,depth in tds])
    ax2.set_ylim([0, None])
    ax2.legend(options)
    fig.tight_layout()
    #plt.show()

if __name__ == '__main__':
    #foo()
    #exit(0)
    print("toy:")
    for d in toy():
        print(d)
    print("therm:")
    for d in therm():
        print(d)
    print("orig:")
    for d in orig():
        print(d)
