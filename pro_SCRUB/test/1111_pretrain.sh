num=(5000 10000 15000 20000 25000 30000 35000 40000 45000)
seed=(1 2 3)

for s in "${seed[@]}";do
	for n in "${num[@]}";do
		python my_main.py --dataset cifar10 --model preactresnet --seed $s --split train --confuse-mode --num-to-forget $n --noise-mode sym --epochs 100 --step-size 50 --gpu 1 --lr 0.1 --save_ True
	done
done

