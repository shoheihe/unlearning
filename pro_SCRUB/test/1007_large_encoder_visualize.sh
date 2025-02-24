num_to_forget_large=(4000 8000 16000 32000)

for num in "${num_to_forget_large[@]}";do
  for i in {1..3}; do
    python after_visualize.py --model 'allcnn' --num-to-forget $num --seed $i --weight-decay 0.0005 --noise-mode asym
    python after_visualize.py --model 'allcnn' --num-to-forget $num --seed $i --weight-decay 0.0005 --noise-mode sym
  done
done    



