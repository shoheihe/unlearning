#python main.py --dataset cifar100 --dataroot=../../data/cifar-100-python --model allcnn --filters 0.4 --lr 0.1 --batch-size 128 --lossfn ce --num-classes 100 --seed 1 --split train --confuse-mode --num-to-forget 4000 --noise-mode sym --epochs 51

python cifar100_unlearning.py --dataset cifar100 --dataroot=../../data/cifar-100-python --model allcnn --lr 0.1 --seed 1 --num-to-forget 4000 --noise-mode sym 
