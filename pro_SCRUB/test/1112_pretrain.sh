num=(10000 20000 5000 15000 25000 30000 35000 40000 45000)
for n in "${num[@]}";do
	python my_main.py --dataset cifar10 --model preactresnet --seed 1 --split train --confuse-mode --num-to-forget $n --noise-mode sym --epochs 100 --step-size 50 --gpu 1 --lr 0.1 --save_ True
done

