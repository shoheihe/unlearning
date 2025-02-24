num_to_forget_small=(2000 4000 8000 16000)

#--resume checkpoints/cifar100_allcnn_1_0_forget_None_lr_0_1_bs_128_ls_ce_wd_0_0005_seed_1_30.pt -

#smallのpretrain
for num in "${num_to_forget_small[@]}";do 
  for i in {1..3}; do
        python main.py --dataset cifar5 --model allcnn --dataroot=data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed $i --split train --confuse-mode --num-to-forget $num --noise-mode asym
        python main.py --dataset cifar5 --model allcnn --dataroot=data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed $i --split train --confuse-mode --num-to-forget $num --noise-mode sym
  done
done
#smallのulearning
for num in "${num_to_forget_small[@]}";do
  for i in {1..3}; do
    python small_unlearning.py --model 'allcnn' --num-to-forget $num --seed $i --weight-decay 0.1 --noise-mode asym
    python small_unlearning.py --model 'allcnn' --num-to-forget $num --seed $i --weight-decay 0.1 --noise-mode sym
  done
done



