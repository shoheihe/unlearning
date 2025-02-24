num=(4000 8000 12000 16000 20000 24000 28000 32000 36000)
seed=(2 3)

python main.py --dataset cifar10 --model allcnn --dataroot=../../data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed 1 --split train --confuse-mode --num-to-forget 36000 --noise-mode sym
for s in "${seed[@]}";do
	for n in "${num[@]}";do
		python main.py --dataset cifar10 --model allcnn --dataroot=../../data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed $s --split train --confuse-mode --num-to-forget $n --noise-mode sym
	done
done
