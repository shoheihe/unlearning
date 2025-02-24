num=(5000 10000 15000 20000 25000 30000 35000 40000 45000)

for n in "{num[@]}";do
	python large_unlearning.py --model preactresnet --num-to-forget $num --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --alpha 0 --delta 3000 --zeta 0.1 --eta 0.1 --method pro
done

python large_unlearning.py --model preactresnet --num-to-forget 20000 --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode asym --alpha 0 --delta 2500 --zeta 0.5 --eta 0.5 --method pro
