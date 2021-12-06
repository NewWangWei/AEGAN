# mle
python train.py --gpus 0,1 --id experiment-mle-newattr-7 --resume_from experiment-mle-newattr-7 --batch_size 64 --learning_rate 3e-4 --sg_data_dir data/sg_data_final_4 --is_limited False --gamma2 0.2

# eval
python eval.py --gpus 5 --resume_from experiment-mle-newattr-7 --batch_size 100 --is_limited False --sg_data_dir data/sg_data_final_4 --beam_size 2

# RL
python train.py --gpus 6,7 --id experiment-rl-newattr-3 --resume_from experiment-mle-newattr-7 --resume_from_best True --learning_rate 2e-5 --batch_size 64 --sg_data_dir data/sg_data_final_4 --is_limited False --self_critical_after 0 --max_epochs 60 --learning_rate_decay_start -1 --scheduled_sampling_start -1 --reduce_on_plateau

# eval
python eval.py --gpus 4 --resume_from experiment-rl-newattr-3 --resume_from_best True --sg_data_dir data/sg_data_final_4 --is_limited False --batch_size 100 --beam_size 2
