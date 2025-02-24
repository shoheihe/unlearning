num=(4000 8000 12000 16000)


for n in "${num[@]}";do
	python main.py --dataset cifar10 --model allcnn --dataroot=../../data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed 1 --split train --confuse-mode --num-to-forget $n --noise-mode sym
	python main.py --dataset cifar10 --model allcnn --dataroot=../../data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed 1 --split train --confuse-mode --num-to-forget $n --noise-mode asym
	python large_unlearning.py --model 'allcnn' --num-to-forget $n --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode asym --alpha 0 --delta 2000 --zeta 0.5 --eta 0.5 --method scrub
	python large_unlearning.py --model 'allcnn' --num-to-forget $n --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --alpha 0 --delta 2000 --zeta 0.5 --eta 0.5 --method scrub
	#python large_unlearning.py --model 'allcnn' --num-to-forget $n --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --alpha 0 --delta 2000 --zeta 0.5 --eta 0.5 --method pro
	python large_unlearning.py --model 'allcnn' --num-to-forget $n --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode asym --alpha 0 --delta 2500 --zeta 0.5 --eta 0.5 --method pro
done
