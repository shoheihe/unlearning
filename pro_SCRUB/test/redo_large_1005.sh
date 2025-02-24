num_to_forget_large=(4000 8000 16000)
num_to_forget_small=(100 200 400 800)

#largeのpretrain
for num in "${num_to_forget_large[@]}";do 
  for i in {1..3}; do
  	python main.py --dataset cifar10 --model allcnn --dataroot=data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed $i --split train --confuse-mode --num-to-forget $num --noise-mode asym
  done
done

#largeのunlearning
for num in "${num_to_forget_large[@]}";do
  for i in {1..3}; do
    python large_unlearning.py --model 'allcnn' --num-to-forget $num --seed $i --weight-decay 0.0005 --noise-mode asym
  done
done    


#largeのpretrain
for num in "${num_to_forget_large[@]}";do 
  for i in {1..3}; do
  	python main.py --dataset cifar10 --model allcnn --dataroot=data/cifar10/ --filters 1.0 --lr 0.01 --weight-decay 0.0005 --batch-size 128 --seed $i --split train --confuse-mode --num-to-forget $num --noise-mode sym
  done
done

#largeのunlearning
for num in "${num_to_forget_large[@]}";do
  for i in {1..3}; do
    python large_unlearning.py --model 'allcnn' --num-to-forget $num --seed $i --weight-decay 0.0005 --noise-mode sym
  done
done    
