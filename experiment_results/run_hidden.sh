num_iters=150000000
dir=experiment_results/raw/hidden_exp_out

cd ..

for i in 0 1 2 3; do
	for j in 1 2 3 4 5; do
		n=$((5 * $i + $j))
		python src/svdplus_exp.py --num-iter $num_iters --train-size 0.88 --verbose hidden-size --start $n --end $n > $dir/hidden_size_$n.out &
		my_pids[$n]=$!

	done
	for pid in ${my_pids[*]}; do
		wait $pid
	done
	unset my_pids
done


#python src/svdplus_exp.py --num-iter $num_iters --train-size 0.88 --verbose hidden-size --start 19 --end 19 > $dir/hidden_size_19.out &
#pid1=$!
#python src/svdplus_exp.py --num-iter $num_iters --train-size 0.88 --verbose hidden-size --start 20 --end 20 > $dir/hidden_size_20.out &
#pid2=$!
#
#wait $pid1
#wait $pid2
