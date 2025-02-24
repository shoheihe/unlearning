
num=(5000 10000 15000 20000)
#num=(30000 35000 40000 45000)
for n in "${num[@]}";do

	python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget $n --seed 1 --pre_train_epoch 200 --noise_mode sym --method scrub --msteps 5 --sgda_epochs 20 --gpu 0 --forget_bs 512 --retain_bs 128 --kd_T 0.5 --eta 0.5
	python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget $n --seed 1 --pre_train_epoch 200 --noise_mode asym --method scrub --msteps 5 --sgda_epochs 20 --gpu 0 --forget_bs 512 --retain_bs 128 --kd_T 0.5 --eta 0.5

done


python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget 25000 --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps 5 --sgda_epochs 20 --gpu 1 --forget_bs 512 --retain_bs 128 --kd_T 0.5 --eta 0.5
