
num_1=(5000 10000 15000 20000)

num_2=(25000 30000 35000 40000 45000)

for n in "${num_1[@]}";do

	python large_unlearning.py --model allcnn --num-to-forget $n --alpha 0 --delta 500 --zeta 20 --eta 1. --noise_mode sym --method scrub --msteps 5 --sgda_epoch 15 --gpu 1 --forget_bs 512 --retain_bs 128 --kd_T 0.5 --file_name allcnn_cifar10_u1.txt --seed 1 --pre_train_epoch 100
	python large_unlearning.py --model allcnn --num-to-forget $n --alpha 0 --delta 500 --zeta 20 --eta 1. --noise_mode asym --method scrub --msteps 5 --sgda_epoch 15 --gpu 1 --forget_bs 512 --retain_bs 128 --kd_T 0.5 --file_name allcnn_cifar10_u1.txt --seed 1 --pre_train_epoch 100

done

for n in "${num_2[@]}";do

	python large_unlearning.py --model allcnn --num-to-forget $n --alpha 0 --delta 500 --zeta 20 --eta 1. --noise_mode sym --method scrub --msteps 5 --sgda_epoch 15 --gpu 1 --forget_bs 512 --retain_bs 128 --kd_T 0.5 --file_name allcnn_cifar10_u1.txt --seed 1 --pre_train_epoch 100
	
done


