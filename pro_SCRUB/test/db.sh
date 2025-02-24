
eps=(0.05 0.01 0.005 0.001)

ms=(50 100 500 1000)

for e in "${eps[@]}";do

	for m in "${ms[@]}";do

		python large_unlearning.py --noise_rate 0.2  --alpha 0 --delta 500 --zeta 20 --eta 1. --noise_mode SDN --method pro --msteps 5 --sgda_epoch 15 --gpu 1 --forget_bs 512 --retain_bs 128 --kd_T 1. --seed 1 --dataset cifar100 --pred k-means --pre_train_epoch 100 --eps $e --min_sample $m

	done
done


