num=(5000 10000 15000 20000 25000 30000 35000 40000 45000)

python large_unlearning.py --model allcnn --num-to-forget 20000 --seed 1 --lr 0.01 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --alpha 0 --delta 2500 --zeta 0.5 --eta 0.5 --method pro --gpu 1 --pre_train_epoch 50 --run ! --tsne 1


for n in "${num[@]}";do
	python my_main.py --dataset cifar10 --model allcnn --seed 1 --split train --confuse-mode --num-to-forget $n --noise-mode sym --epochs 100 --step-size 50 --gpu 1 --lr 0.01 --filters 1 --save_ True
	python large_unlearning.py --model allcnn --num-to-forget $n --seed 1 --lr 0.01 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --alpha 0 --delta 2500 --zeta 0.5 --eta 0.5 --method pro --gpu 1 --pre_train_epoch 100
done

