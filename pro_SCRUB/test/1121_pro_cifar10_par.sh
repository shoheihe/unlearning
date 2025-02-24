
delta=(500 1000)

zeta=(10 20)

python large_unlearning.py --num-to-forget 25000 --alpha 0 --delta 500 --zeta 10 --eta 1. --noise_mode sym --method pro --msteps 5 --sgda_epoch 15 --gpu 1 --forget_bs 512 --retain_bs 128 --kd_T 0.5 --file_name pro_cifar10_parameter.txt

python large_unlearning.py --num-to-forget 25000 --alpha 0 --delta 500 --zeta 20 --eta 1. --noise_mode sym --method pro --msteps 5 --sgda_epoch 15 --gpu 1 --forget_bs 512 --retain_bs 128 --kd_T 0.5 --file_name pro_cifar10_parameter.txt

python large_unlearning.py --num-to-forget 25000 --alpha 0 --delta 1000 --zeta 20 --eta 1. --noise_mode sym --method pro --msteps 5 --sgda_epoch 15 --gpu 1 --forget_bs 512 --retain_bs 128 --kd_T 0.5 --file_name pro_cifar10_parameter.txt

#python large_unlearning.py --num-to-forget 25000 --alpha 0 --delta $d --zeta $z --eta 1. --noise_mode sym --method pro --msteps 5 --sgda_epoch 15 --gpu 1 --forget_bs 512 --retain_bs 128 --kd_T 0.5 --file_name pro_cifar10_parameter.txt


