num=(10000 15000 20000 25000 30000 35000 40000 45000)

for n in "${num[@]}";do
	python large_unlearning.py --model preactresnet --dataset cifar100 --num-to-forget $n --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps 4 --sgda_epochs 10 --gpu 0 --forget_bs 256 --retain_bs 256 --kd_T 0.1
done
