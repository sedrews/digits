import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as be_pdf

from dataparser import RunStats

# Plot total time as a function of depth
def timevsdepth(runs, names):
    for run in runs:
        xs = []
        ys = []
        for s in run.synthstats:
            xs.append(s.depth)
            ys.append(s.time)
        plt.plot(xs,ys)
    plt.legend(names)
    plt.ylabel("total time (s)")
    plt.xlabel("depth")
    plt.axis([0, None, 0, 600])

# Plot best solution as a function of time
def bestvstime(runs, names):
    for run in runs:
        xs = []
        ys = []
        for s in run.beststats:
            xs.append(s.time)
            ys.append(s.stats.error)
        plt.plot(xs,ys,'o-')
    plt.legend(names)
    plt.ylabel("best error")
    plt.xlabel("total time (s)")
    plt.xscale('log')
    plt.axis([0, 600, 0, 1])

# Plot best solution as a function of depth
def bestvsdepth(runs, names):
    for run in runs:
        xs = []
        ys = []
        for s in run.beststats:
            xs.append(s.depth)
            ys.append(s.stats.error)
        plt.plot(xs,ys,'o-')
    plt.legend(names)
    plt.ylabel("best error")
    plt.xlabel("depth")
    plt.axis([0, None, 0, 1])

# Plot the number of synthesizer calls as a function of depth
def synthcallsvsdepth(runs, names):
    for run in runs:
        xs = []
        ys = []
        for s in run.synthstats:
            xs.append(s.depth)
            ys.append(s.stats.calls)
        plt.plot(xs,ys)
    plt.legend(names)
    plt.ylabel("synthesizer calls")
    plt.xlabel("depth")
    plt.axis([0, None, 0, None])

# Plot the number of synthesizer calls as a function of time
def synthcallsvstime(runs, names):
    for run in runs:
        xs = []
        ys = []
        for s in run.synthstats:
            xs.append(s.time)
            ys.append(s.stats.calls)
        plt.plot(xs,ys)
    plt.legend(names)
    plt.ylabel("synthesizer calls")
    plt.xlabel("total time (s)")
    plt.axis([0, 600, 0, None])

# The growth of the sampleset as a function of depth
def growthvsdepth(runs, names):
    for run in runs:
        xs = []
        ys = []
        for s in run.synthstats:
            xs.append(s.depth)
            ys.append(s.stats.sat)
        plt.plot(xs,ys)
    plt.legend(names)
    plt.ylabel("growth (total satisfiable synthesis queries)")
    plt.xlabel("depth")
    plt.axis([0, None, 0, None])

def plotbench(title, prefix, vals, suffix):
    fnames = [prefix + val + suffix for val in vals]
    runs = [RunStats(f) for f in fnames]
    plt.subplot(2,3,1)
    timevsdepth(runs, vals)
    plt.subplot(2,3,2)
    bestvstime(runs, vals)
    plt.subplot(2,3,3)
    bestvsdepth(runs, vals)
    plt.subplot(2,3,5)
    synthcallsvstime(runs, vals)
    plt.subplot(2,3,6)
    synthcallsvsdepth(runs, vals)
    # Growth function only needs the -o 1 version
    plt.subplot(2,3,4)
    growthvsdepth([runs[0]],[vals[0]])
    plt.suptitle(title)

if __name__ == '__main__':
    titles = ["dt4", "dt16", "dt44", \
              "svm3-1", "svm3-2", "svm3-3", "svm3-4", \
              "svm4-1", "svm4-2", "svm4-3", "svm4-4", "svm4-5", \
              "svm5-1", "svm5-2", "svm5-3", "svm5-4", "svm5-5", "svm5-6"]
    i = 0
    for title in titles:
        print("building plot for", title)
        plt.figure(num=i, figsize=(16,10))
        prefix = "orig/out/" + title + "_s0_"
        suffix = ".out"
        vals = ["o1", "o.5", "o.375", "o.25", "o.125", "o.0625", "a1..03"]
        plotbench(title, prefix, vals, suffix)
        plt.tight_layout(pad=3)
        i += 1
    #plt.show()
    print("writing to output.pdf")
    pdf = be_pdf.PdfPages("output.pdf")
    for n in range(i):
        pdf.savefig(plt.figure(n))
    pdf.close()
