f=$1
inc=( 1 2 4 6 8 12 16 24 )

for b in "${inc[@]}"
do
    let s=f*b
    python test_nums.py --features "$f" --block-size "$s" --use-head --cluster-shape=2,1
    sleep 2
done