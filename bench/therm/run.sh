RESDIR=$1
CVC4=$2
mkdir $RESDIR
if [ $? -ne 0 ]; then
    echo Specify a non-existing directory to create and store output files
    exit
fi
PARAMS="-d 1000 -t 600 -a 1 0 -s 0 --fspec \"lambda *args : 0\""
echo $PARAMS
if [ -z $CVC4 ]; then
    echo "no CVC4 path provided; using z3"
else
    echo "using CVC4 path at $CVC4"
    PARAMS="$PARAMS --cvc4 $CVC4"
fi
echo $PARAMS
for F in $(ls fr); do
    FN=${F%.fr} # Strip the file extension
    OUT="${RESDIR}/${FN}.jsonl"
    CMD="python run.py -f fr/$F $PARAMS -j $OUT"
    echo $(date): $CMD
    bash -c "$CMD"
done
