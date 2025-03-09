rs=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
models=('preactresnet18' 'allcnn')
datasets=('cifar10' 'cifar100')

for r in "${rs[@]}";do
    for m in "${models[@]}";do
        for d in "${datasets[@]}";do

            python main.py --dataset $d --model $m --noise_rate $r --noise_mode sym --save True --tsne True

        done
    done
done

rs=(0.1 0.2 0.3 0.4)

for r in "${rs[@]}";do
    for m in "${models[@]}";do

        python main.py --dataset cifar10 --model $m --noise_rate $r --noise_mode asym --save True --tsne True

    done
done

python main.py --dataset cifar100 --model allcnn --noise_rate 0.4 --noise_mode asym --save True --tsne True

python main.py --dataset cifar100 --model preactresnet18 --noise_rate 0.4 --noise_mode asym --save True --tsne True

