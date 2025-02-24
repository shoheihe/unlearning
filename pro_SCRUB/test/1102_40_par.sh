zeta=(0.2 0.8)
delta=(2000 1250)
eta=(0.5 0.8 1.0) 

for z in "${zeta[@]}";do
	for e in "${eta[@]}";do
		for d in "${delta[@]}";do
			python large_unlearning.py --model 'allcnn' --num-to-forget 16000 --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --alpha 0 --delta $d --zeta $z --eta $e --method pro
		done
	done
done
