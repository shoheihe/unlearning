num_to_forget_small=(2000 4000 8000 16000)

#--resume checkpoints/cifar100_allcnn_1_0_forget_None_lr_0_1_bs_128_ls_ce_wd_0_0005_seed_1_30.pt 
#smallのulearning
for num in "${num_to_forget_small[@]}";do
  for i in {1..3}; do
    python small_unlearning.py --model 'allcnn' --dataset cifar5 --num-to-forget $num --seed $i --weight-decay 0.0005 --noise-mode asym
    python small_unlearning.py --model 'allcnn' --dataset cifar5 --num-to-forget $num --seed $i --weight-decay 0.0005 --noise-mode sym
  done
done



