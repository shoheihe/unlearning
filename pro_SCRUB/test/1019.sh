deltas=(1 5 10)
zetas=(1 2 3)

for d in "${deltas[@]}";do
	for z in "${zetas[@]}";do
		python large_unlearning.py --model 'allcnn' --num-to-forget 8000 --seed 1 --weight-decay 0.0005 --delta $d --zeta $z --noise-mode sym --method pro
	done
done
 
