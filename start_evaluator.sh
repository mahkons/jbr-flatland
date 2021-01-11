
for arg in "$@"
do
    case "$arg" in
        "clean")
            redis-cli shutdown
            rm -rf dump.rdb
        ;;
        "small"|"medium"|"big"|"tiny"|"2")
            testdir="./scratch/test-envs-$arg"
            echo "$testdir"
        ;;
   esac
done

redis-server &
flatland-evaluator --tests="$testdir" --shuffle=False
