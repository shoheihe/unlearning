
lr=(500 700 1000)

per=(30 40 50 100)

for l in "${lr[@]}";do
	for p in "${per[@]}";do
	
       		python visualize.py --model preactresnet --dataset cifar10 --num-to-forget 25000 --seed 1 --pre_train_epoch 200 --noise_mode sym --method pro --msteps 5 --sgda_epochs 15 --gpu 0 --forget_bs 512 --retain_bs 128 --kd_T 2. --delta 500 --zeta 20 --eta 0.5 --alpha 0. --tsne_lr $l --tsne_per $p	 
	 
	 done
 done

