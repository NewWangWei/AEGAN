# AGAN

## Models

The trained Captioner Model and data can be found in <a href="https://drive.google.com/drive/folders/11q-XixaukQzjJ87kbhUQQxzkSRrigld7?usp=sharing" target="_blank">google drive</a>.

## Scripts

### MLE

```python train.py --gpus 0,1 --id experiment-mle-newattr --resume_from experiment-mle-newattr-7 --batch_size 64 --learning_rate 3e-4 --sg_data_dir data/sg_data_final_4 --is_limited False --gamma2 0.2```

## RL

```python train.py --gpus 0,1 --id experiment-rl-newattr --resume_from experiment-mle-newattr --resume_from_best True --learning_rate 2e-5 --batch_size 64 --sg_data_dir data/sg_data_final_4 --is_limited False --self_critical_after 0 --max_epochs 60 --learning_rate_decay_start -1 --scheduled_sampling_start -1 --reduce_on_plateau```

## Eval

```python eval.py --gpus 0 --resume_from experiment-mle-newattr --batch_size 100 --is_limited False --sg_data_dir data/sg_data_final_4 --beam_size 2```

or

```python eval.py --gpus 0 --resume_from experiment-rl-newattr --resume_from_best True --sg_data_dir data/sg_data_final_4 --is_limited False --batch_size 100 --beam_size 2```







