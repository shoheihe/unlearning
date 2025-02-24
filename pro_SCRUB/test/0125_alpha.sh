
seed=(1 2 3 4 5 6 7 8 9 10)
file_name="50_alpha.txt"

for s in "${seed[@]}";do

	python large_unlearning.py --model preactresnet --dataset cifar10 --noise_rate 0.5  --seed $s --pre_train_epoch 200 --noise_mode sym --method pro --gpu 1 --file_name $file_name --delta 500 --eta 0.5 --zeta 20 --kd_T 0.5 --alpha 0 --forget_bs 512 --retain_bs 128
	python large_unlearning.py --model preactresnet --dataset cifar10 --noise_rate 0.5  --seed $s --pre_train_epoch 200 --noise_mode sym --method pro --gpu 1 --file_name $file_name --delta 500 --eta 0.5 --zeta 20 --kd_T 0.5 --alpha 1 --forget_bs 512 --retain_bs 128

done

