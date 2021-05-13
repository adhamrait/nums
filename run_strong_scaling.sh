cpus=( 1 2 4 8 16 32 )
for c in "${cpus[@]}"
do
    ./run_all_iter.sh "$c"
done