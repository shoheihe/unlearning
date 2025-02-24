	
num_1=(5000 10000 15000 20000)

num_2=(25000 30000 35000 40000 45000)

for n in "${num_1[@]}";do

	python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget $n --seed 1 --pre_train_epoch 200 --noise_mode sym --method scrub --msteps 2 --sgda_epochs 15 --gpu 0 --forget_bs 512 --retain_bs 128 --pred gmm --kd_T 2.

python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget $n --seed 1 --pre_train_epoch 200 --noise_mode asym --method scrub --msteps 2 --sgda_epochs 15 --gpu 0 --forget_bs 512 --retain_bs 128 --pred gmm --kd_T 2.

done

for n in "${num_2[@]}";do

        python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget $n --seed 1 --pre_train_epoch 200 --noise_mode sym --method scrub --msteps 2 --sgda_epochs 15 --gpu 0 --forget_bs 512 --retain_bs 128 --pred gmm --kd_T 2.

done

