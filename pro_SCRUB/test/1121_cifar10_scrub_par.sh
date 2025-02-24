f_bs=(128 256 512 1024)
r_bs=(128 256)

kds=(2 0.5 0.1)

for f in "${f_bs[@]}";do
	for r in "${r_bs[@]}";do
		for kd in "${kds[@]}";do
			python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget 25000 --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps 2 --sgda_epochs 10 --gpu 1 --forget_bs $f --retain_bs $r --kd_T $kd
		done
	done
done
