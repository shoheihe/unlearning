rs=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
models=('preactresnet18' 'allcnn')

for r in "${rs[@]}";do
    for m in "${models[@]}";do

        python unlearning.py --dataset cifar10  --model $m --noise_rate $r --noise_mode asym  --forget_bs 1024

        python unlearning.py --dataset cifar100 --model $m --noise_rate $r --noise_mode asym --forget_bs 1024

    done
done

rs=(0.1 0.2 0.3 0.4)

for r in "${rs[@]}";do
    for m in "${models[@]}";do

        python unlearning.py --dataset cifar10  --model $m --noise_rate $r --noise_mode asym  --forget_bs 1024

    done
done

python unlearning.py --dataset cifar100  --model allcnn --noise_rate 0.4 --noise_mode asym  --forget_bs 1024
python unlearning.py --dataset cifar100  --model preactresnet18 --noise_rate 0.4 --noise_mode asym  --forget_bs 1024
