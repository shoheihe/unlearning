seed=(12 13 14 15)
for s in "${seed[@]}";do
	python main.py --dataset cifar10 --model allcnn --dataroot=../../data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed $s --split train --confuse-mode --num-to-forget 8000 --noise-mode sym 
	python large_unlearning.py --model 'allcnn' --num-to-forget 8000 --seed $s --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --delta 250 --zeta 10 --method pro
	python large_unlearning.py --model 'allcnn' --num-to-forget 8000 --seed $s --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --alpha 0 --delta 250 --zeta 10 --method pro
done
