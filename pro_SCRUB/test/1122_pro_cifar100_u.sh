


num_1=(5000 10000 15000 20000 25000 30000 35000 40000 45000)

for n in "${num_1[@]}";do

	python large_unlearning.py --model preactresnet --dataset cifar100 --num-to-forget $n --seed 1 --pre_train_epoch 200 --noise_mode sym --method pro --msteps 5 --sgda_epochs 15 --gpu 0 --forget_bs 1024 --retain_bs 128 --kd_T 0.5 --delta 500 --zeta 5 --eta 1
done

