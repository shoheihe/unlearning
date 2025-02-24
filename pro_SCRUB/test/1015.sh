seed=(1 2 3)
delta=(0 100 250 500)
zeta=(0 10 50 100)

for s in "${seed[@]}";do
  for d in "${delta[@]}";do
    for z in "${zeta[@]}";do
	python large_unlearning.py --model 'allcnn' --num-to-forget 16000 --seed $s --weight-decay 0.0005 --sgda_epochs 15 --delta $d --zeta $z --noise-mode asym --method pro
	python large_unlearning.py --model 'allcnn' --num-to-forget 16000 --seed $s --weight-decay 0.0005 --sgda_epochs 15 --delta $d --zeta $z --noise-mode sym --method pro
    done
  done
done
