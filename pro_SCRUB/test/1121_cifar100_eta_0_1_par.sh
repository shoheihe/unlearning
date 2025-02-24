msteps=(2 3 5)


for m in "${msteps[@]}";do
			python large_unlearning.py --model preactresnet --dataset cifar100 --num-to-forget 25000 --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps $m --sgda_epochs 20 --gpu 0 --forget_bs 512 --retain_bs 128 --kd_T 0.5	
done

