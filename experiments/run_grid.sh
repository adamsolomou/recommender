#num_iters=150000000
num_iters=150
dir=grid_exp_out
point_num=30
sleep_time=3
train_size=0.88

# begin[0]="0.01"
# begin[1]="0.05"
# begin[2]="0.09"
# begin[3]="0.13"
# begin[4]="0.17"
# 
# end[0]="0.04"
# end[1]="0.08"
# end[2]="0.12"
# end[3]="0.16"
# end[4]="0.2"

for j in 7 8 9 10 11; do
	python src/svdplus_exp.py --num-iter $num_iters --train-size $train_size --sleep $sleep_time --verbose grid --hidden-start $j --hidden-end $j --reg-start 0.07 --reg-end 0.11 --reg-point-num $point_num > $dir/grid_$j.out &
	my_pids[$j]=$!

done

for pid in ${my_pids[*]}; do
	wait $pid
	echo $pid returned
done
