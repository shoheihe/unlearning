num=(10000 25000 5000 15000 20000 30000 35000 40000 45000)


python my_main.py --gpu 0 --epochs 100 --dataset cifar10 --batch-size 128 --lr 0.01 --seed 1 --num-to-forget 20000 --step-size 50
for n in "${num[@]}";do
	python my_main.py --gpu 0 --epochs 100 --noise-mode sym --dataset cifar10 --batch-size 128 --lr 0.01 --seed 1 --num-to-forget $n --step-size 50
done

