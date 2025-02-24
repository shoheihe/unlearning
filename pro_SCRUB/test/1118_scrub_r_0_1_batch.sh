python large_unlearning.py --model preactresnet --dataset cifar100 --num-to-forget 5000 --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps 4 --sgda_epochs 10 --gpu 1 --forget_bs 256 --retain_bs 128

python large_unlearning.py --model preactresnet --dataset cifar100 --num-to-forget 5000 --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps 4 --sgda_epochs 10 --gpu 1 --forget_bs 512 --retain_bs 32


#cifar10 in paper
python large_unlearning.py --model preactresnet --dataset cifar100 --num-to-forget 5000 --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps 4 --sgda_epochs 10 --gpu 1 --forget_bs 512 --retain_bs 128

python large_unlearning.py --model preactresnet --dataset cifar100 --num-to-forget 5000 --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps 4 --sgda_epochs 10 --gpu 1 --forget_bs 512 --retain_bs 256

python large_unlearning.py --model preactresnet --dataset cifar100 --num-to-forget 5000 --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps 4 --sgda_epochs 10 --gpu 1 --forget_bs 512 --retain_bs 512

python large_unlearning.py --model preactresnet --dataset cifar100 --num-to-forget 5000 --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps 4 --sgda_epochs 10 --gpu 1 --forget_bs 1024 --retain_bs 128

python large_unlearning.py --model preactresnet --dataset cifar100 --num-to-forget 5000 --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps 4 --sgda_epochs 10 --gpu 1 --forget_bs 1024 --retain_bs 256

python large_unlearning.py --model preactresnet --dataset cifar100 --num-to-forget 5000 --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps 4 --sgda_epochs 10 --gpu 1 --forget_bs 1024 --retain_bs 512

python large_unlearning.py --model preactresnet --dataset cifar100 --num-to-forget 5000 --seed 1 --pre_train_epoch 200 --noise-mode sym --method scrub --msteps 4 --sgda_epochs 10 --gpu 1 --forget_bs 1024 --retain_bs 1024


