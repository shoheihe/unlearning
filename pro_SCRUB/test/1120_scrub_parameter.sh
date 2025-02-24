mstep=(2 3 5)
for m in "${mstep[@]}";do
	python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget 25000 --seed 1 --gamma 1 --alpha 0 --delta 100 --zeta 0.5 --eta 0.5 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps $m --sgda_epochs 20 --gpu 1 --forget_bs 128 --retain_bs 128 --kd_T 0.5		
	python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget 25000 --seed 1 --gamma 1 --alpha 0 --delta 100 --zeta 0.5 --eta 0.5 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps $m --sgda_epochs 20 --gpu 1 --forget_bs 512 --retain_bs 128 --kd_T 2.
done
