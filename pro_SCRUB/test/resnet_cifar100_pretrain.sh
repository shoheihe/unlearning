num=(5000 10000 15000 20000 25000 30000 35000 40000 45000)

for n in "${num[@]}";do
	python my_main.py --dataset cifar100 --model preactresnet --epochs 200 --num-to-forget $n --seed 1 --step-size 100 --noise-mode sym --gpu 0 --save True
	python my_main.py --dataset cifar100 --model preactresnet --epochs 200 --num-to-forget $n --seed 1 --step-size 100 --noise-mode asym --gpu 0 --save True
done

