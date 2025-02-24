num=(5000 10000 15000 20000 25000 30000 35000 40000 45000)
seed=(1 2 3)

for s in "${seed[@]}";do
	for n in "${num[@]}";do
		python main.py --dataset cifar10 --model reactresnet --dataroot=../../data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed $s --split train --confuse-mode --num-to-forget $n --noise-mode sym
	done
	python main.py --dataset cifar10 --model reactresnet --dataroot=../../data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed $s --split train --confuse-mode --num-to-forget 20000 --noise-mode asym
done
