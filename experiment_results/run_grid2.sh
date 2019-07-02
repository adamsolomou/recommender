## Number of iterations for the neural network.
num_iters=150000000

# The directory where the output will be stores. If it doesn't exist you need to create it.
dir=experiment_results/raw/grid2_exp_out

# How many points along the lambda1 axis each process will calculate
point_num=5

# Leave at 0. I used it to let my pc cooldown when running experiments on it.
sleep_time=0

# The ratio of the training set to the whole data set.
train_size=0.88

# Starting and ending points for each process on the lambda1 axis.
begin[0]="0.07"
begin[1]="0.075"
begin[2]="0.08"
begin[3]="0.085"
begin[4]="0.09"
begin[5]="0.095"
begin[6]="0.1"
begin[7]="0.105"

end[0]="0.074"
end[1]="0.079"
end[2]="0.084"
end[3]="0.089"
end[4]="0.094"
end[5]="0.099"
end[6]="0.104"
end[7]="0.11"

my_pids=()

cd ..

# Adjust j and i ranges to run for different values
for j in 10 11; do
	for i in 0 1; do
		python src/svdplus_exp.py --num-iter $num_iters --train-size $train_size --sleep $sleep_time --verbose grid --hidden-start $j --hidden-end $j --reg-start ${begin[$i]} --reg-end ${end[$i]} --reg-point-num $point_num > $dir/grid2_${j}_$i.out &
		my_pids+="$! "
	done

done

for pid in ${my_pids[*]}; do
	wait $pid
	echo $pid returned
done
