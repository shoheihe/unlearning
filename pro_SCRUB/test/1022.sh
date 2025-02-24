delta=(5 10 50 100)
zeta=(5 10)

for d in  "${delta[@]}";do
	for z in "${zeta[@]}";do 
		python large_unlearning.py --model 'allcnn' --num-to-forget 4000 --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --delta $d --zeta $z --method pro
	done
done
