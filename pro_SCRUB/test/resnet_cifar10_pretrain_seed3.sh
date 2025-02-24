
num_1=(15000 20000)

num_2=(25000 30000 35000 40000 45000)

for n in "${num_1[@]}";do

	python my_main.py --dataset cifar10 --model preactresnet --epochs 200 --num-to-forget $n --seed 3 --step-size 100 --noise-mode sym --gpu 0 --save True --tsne True
	
	python my_main.py --dataset cifar10 --model preactresnet --epochs 200 --num-to-forget $n --seed 3 --step-size 100 --noise-mode asym --gpu 0 --save True --tsne True

done

for n in "${num_2[@]}";do

	python my_main.py --dataset cifar10 --model preactresnet --epochs 200 --num-to-forget $n --seed 3 --step-size 100 --noise-mode sym --gpu 0 --save True --tsne True

done

