#pro only
#python large_unlearning.py --model 'allcnn' --num-to-forget 8000 --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --alpha 0 --beta 0 --gamma 0 --eta 0  --delta 250 --zeta 10 --method pro

#best performance
#python large_unlearning.py --model 'allcnn' --num-to-forget 8000 --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --delta 250 --zeta 10 --method pro
#zeta*10
#python large_unlearning.py --model 'allcnn' --num-to-forget 8000 --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --delta 250 --zeta 100 --method pro
#*10
#python large_unlearning.py --model 'allcnn' --num-to-forget 8000 --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --delta 2500 --zeta 100 --method pro
#*5
#python large_unlearning.py --model 'allcnn' --num-to-forget 8000 --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --delta 1250 --zeta 50 --method pro
#40% do best of 20% 
#python large_unlearning.py --model 'allcnn' --num-to-forget 16000 --seed 1 --weight-decay 0.0005 --msteps 2 --sgda_epochs 15 --noise-mode sym --delta 250 --zeta 10 --method pro




zeta=(1 5 10 50 100)
delta=(1 2 3 4 5)

for z in "${zeta[@]}";do
	for d in "${delta[@]}";do
		python small_unlearning.py --model 'allcnn' --num-to-forget 100 --seed 1 --weight-decay 0.1 --msteps 2 --sgda_epochs 10 --noise-mode sym --method pro --gamma 15 --alpha 0 --beta 0 --delta $d --zeta $z --eta 50
	done
done


