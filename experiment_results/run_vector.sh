num_iters=150000000
dir=experiment_results/raw/vector_exp_out
point_num=4
sleep_time=0
train_size=0.88

begin[0]="0.01"
begin[1]="0.05"
begin[2]="0.09"
begin[3]="0.13"
begin[4]="0.17"

end[0]="0.04"
end[1]="0.08"
end[2]="0.12"
end[3]="0.16"
end[4]="0.2"

cd ..

for j in 0 1 2 3 4; do
	python src/svdplus_exp.py --num-iter $num_iters --train-size $train_size --sleep $sleep_time --verbose reg-vector --start ${begin[$j]} --end ${end[$j]} --point-num $point_num > $dir/vector_reg_$j.out &
	my_pids[$j]=$!

done

for pid in ${my_pids[*]}; do
	wait $pid
	echo $pid returned
done
# unset my_pids
# sleep $sleep_time


#python src/svdplus_exp.py --num-iter $num_iters --train-size 0.88 --verbose hidden-size --start 19 --end 19 > $dir/hidden_size_19.out &
#pid1=$!
#python src/svdplus_exp.py --num-iter $num_iters --train-size 0.88 --verbose hidden-size --start 20 --end 20 > $dir/hidden_size_20.out &
#pid2=$!
#
#wait $pid1
#wait $pid2
