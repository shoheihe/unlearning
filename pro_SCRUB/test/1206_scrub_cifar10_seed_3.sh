
num_1=(5000 10000 15000 20000)

num_2=(25000 30000 35000 40000 45000)

file_name="scrub_cifar10_seed_3"

for n in "${num_1[@]}";do

	python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget $n --seed 3 --pre_train_epoch 200 --noise_mode sym --method scrub --gpu 1 --file_name $file_name
	python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget $n --seed 3 --pre_train_epoch 200 --noise_mode asym --method scrub --gpu 1 --file_name $file_name

done

for n in "${num_2[@]}";do

	python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget $n --seed 3 --pre_train_epoch 200 --noise_mode sym --method scrub --gpu 1  --file_name $file_name

done

