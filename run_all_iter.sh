sizes=( 128 256 512 1024 1536 2048 3072 )
for s in "${sizes[@]}"
do
    ./run_one_iter.sh "$s"
done