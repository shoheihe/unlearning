delta=(20 50 100 150 200 250)
zeta=(1 2 3 5 7 9 10)
num_to_forget=(100 200 400)

for d in "${delta[@]}";do
	for z in "${zeta[@]}";do
		for num in "${num_to_forget[@]}";do
			python small_unlearning.py --model 'allcnn' --num-to-forget $num --seed 1 --weight-decay 0.1 --msteps 5 --sgda_epochs 15 --delta $d --zeta $z --noise-mode sym --method pro
		done
	done
done
