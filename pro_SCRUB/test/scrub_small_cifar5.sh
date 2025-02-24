


seeds=(1 2 3)
nums=(100 200 400)
modes=(sym asym)

for s in "${seeds[@]}";do
	for num in "${nums[@]}";do
		for mode in "${modes[@]}";do
			python small_unlearning.py --model 'allcnn' --num-to-forget $num --seed $s --weight-decay 0.1 --msteps 2 --sgda_epochs 10 --noise-mode $mode --method scrub
                done
        done
done

