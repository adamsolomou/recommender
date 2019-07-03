num_iters=150000000
dir=experiment_results/raw/hidden_exp_out

cd ../..
echo $(pwd)

for i in 0 1 2 3; do
	for j in 1 2 3 4 5; do
		n=$((5 * $i + $j))
		python src/experiments/svdplus_exp.py --num-iter $num_iters --train-size 0.88 --verbose hidden-size --start $n --end $n > $dir/hidden_size_$n.out &
		my_pids[$n]=$!

	done
	for pid in ${my_pids[*]}; do
		wait $pid
	done
	unset my_pids
done
