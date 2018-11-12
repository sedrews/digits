FILELIST=$1 # A list of paths to .fr files
PARAMLIST=$2 # A list of arguments --- first block is a filename identifier
RESDIR=$3
mkdir $RESDIR
if [ $? -ne 0 ]; then
    echo Specify a non-existing directory to create and store output files
    exit
fi
for F in $(cat $FILELIST); do
    while read P; do
        FN=${F%.fr} # Strip the file extension
        FN=${FN##*/} # Strip the leading path
        ID=$(echo $P | sed 's/\s.*$//') # Isolate the block before the first space
        ARGS=$(echo $P | sed 's/^[^ ]* *//') # Isolate the remaining argument list
        OUT="${RESDIR}/${FN}${ID}.jsonl"
        CMD="python run.py -f $F $ARGS -j $OUT"
        echo "$(date): $CMD"
        $CMD
    done < $PARAMLIST
done
