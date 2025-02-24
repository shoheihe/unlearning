python my_main.py --gpu 0 --epochs 100 --dataset cifar10 --batch-size 128 --lr 0.01 --seed 1 --num-to-forget 20000 --step-size 50 --model preaxtresnet --noise-mode asym

python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget 20000 --seed 1 --gamma 1 --alpha 0 --delta 2500 --zeta 0.5 --eta 0.5 --pre_train_epoch 100 --noise-mode asym --method pro --msteps 2 --sgda_epochs 15 --gpu 0

num=(5000 10000 15000 20000 25000 30000)
for n in "${num[@]}";do
	python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget $n --seed 1 --gamma 1 --alpha 0 --delta 2500 --zeta 0.5 --eta 0.5 --pre_train_epoch 100 --noise-mode sym --method pro --msteps 2 --sgda_epochs 15 --gpu 0
done

python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget 35000 --seed 1 --gamma 1 --alpha 0 --delta 2500 --zeta 0.2 --eta 0.2 --pre_train_epoch 100 --noise-mode sym --method pro --msteps 2 --sgda_epochs 15 --gpu 0
python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget 40000 --seed 1 --gamma 1 --alpha 0 --delta 3000 --zeta 0.2 --eta 0.2 --pre_train_epoch 100 --noise-mode sym --method pro --msteps 2 --sgda_epochs 15 --gpu 0
python large_unlearning.py --model preactresnet --dataset cifar10 --num-to-forget 45000 --seed 1 --gamma 1 --alpha 0 --delta 3000 --zeta 0.2 --eta 0.2 --pre_train_epoch 100 --noise-mode sym --method pro --msteps 2 --sgda_epochs 15 --gpu 0

