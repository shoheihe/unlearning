seed=(2 3)

for s in "${seed[@]}";do
	python large_unlearning.py --model 'allcnn' --num-to-forget 4000 --seed $s --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --alpha 0 --delta 250 --zeta 10 --method pro
	python large_unlearning.py --model 'allcnn' --num-to-forget 4000 --seed $s --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode asym --alpha 0 --delta 250 --zeta 10 --method pro
done
