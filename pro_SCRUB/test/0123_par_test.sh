seed=(4 5 6 7 8 9 10)

for s in "${seed[@]}";do

	python my_main.py --dataset cifar10 --model preactresnet --epochs 200 --noise_rate 0.5  --seed $s --step-size 100 --noise_mode sym --gpu 1

done










#python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget $n --seed 3 --pre_train_epoch 200 --noise_mode sym --methb --gpu 1 --file_name $file_name
