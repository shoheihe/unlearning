zeta=(1 3 5)
delta=(1250 1000 500)
eta=(0.2 0.5 1)

for z in "${zeta[@]}";do
	for e in "${eta[@]}";do
		for d in "${delta[@]}";do
	#python main.py --dataset cifar10 --model allcnn --dataroot=../../data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed $s --split train --confuse-mode --num-to-forget 12000 --noise-mode sym
	#python main.py --dataset cifar10 --model allcnn --dataroot=../../data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed $s --split train --confuse-mode --num-to-forget 12000 --noise-mode asym
			python large_unlearning.py --model 'allcnn' --num-to-forget 12000 --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --alpha 0 --delta $d --zeta $z --eta $e --method pro
			python large_unlearning.py --model 'allcnn' --num-to-forget 12000 --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode asym --alpha 0 --delta $d --zeta $z --eta $ e--method pro
		done
	done
done
