



zeta=(0 1 10 100 500)

for z in "${zeta[@]}";do

	python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget 45000 --seed 1 --pre_train_epoch 200 --noise_mode sym --method pro --gpu 0 --kd_T 0.5 --alpha 0 --eta 0.5 --zeta $z  --delta 500 --mstep 5 --sgda_epochs 15 --forget_bs 512
done



