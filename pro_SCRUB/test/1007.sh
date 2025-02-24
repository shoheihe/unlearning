num_to_forget_small=(100 200 400 450)


#smallのulearning
for num in "${num_to_forget_small[@]}";do
  for i in {1..3}; do
    python small_unlearning.py --model 'allcnn' --num-to-forget $num --seed $i --weight-decay 0.1 --noise-mode asym
    python small_unlearning.py --model 'allcnn' --num-to-forget $num --seed $i --weight-decay 0.1 --noise-mode sym
  done
done


num_to_forget_large=(4000 8000 16000 32000)
#largeのunlearning
for num in "${num_to_forget_large[@]}";do
  for i in {1..3}; do
    python large_unlearning.py --model 'allcnn' --num-to-forget $num --seed $i --weight-decay 0.0005 --noise-mode asym
    python large_unlearning.py --model 'allcnn' --num-to-forget $num --seed $i --weight-decay 0.0005 --noise-mode sym
  done
done    


