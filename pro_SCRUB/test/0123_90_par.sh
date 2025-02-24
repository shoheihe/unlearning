
delta=(100 500 2500)

eta=(0.5 1 5) 

zeta=(5 20 100)

for e in "${eta[@]}";do

	for d in "${delta[@]}";do

		for z in "${zeta[@]}";do

			python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget 45000 --seed 1 --pre_train_epoch 200 --noise_mode sym --method pro --gpu 0 --file_name "90_par.txt" --kd_T 0.5 --alpha 0 --eta $e --zeta $z --delta $d --mstep 5 --sgda_epochs 15 --forget_bs 512
		done
	done
done

