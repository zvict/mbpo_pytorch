#!/bin/bash

# source ~/.bashrc

# export PATH="/home/yza629/anaconda3/envs/mbpo/bin/:$PATH"
# module load cuda/11.1.1.lua

# git checkout exp-ys
# module load httpproxy

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yza629/.mujoco/mjpro150/bin

nvidia-smi

# eps0.4-ga_lr0.05-iter40-n_critic5-end_iter50-eps_decay_end100000-noisy5-eff0


job=0
eps=0.0
iter=40
n_critic=5
end_iter=50
ga_lr=0.01
eps_decay_end=100000
noisy_coef=4
efficient=1

for seed in 0; do
    name=eps${eps}-seed${seed}
    setting=eps${eps}
    project_name=mbpo-walker-ys
    save_dir=/NAS/yza629/codes/mbpo/results/${project_name}
    mkdir -p ${save_dir}/${name}
    save_log_file=${save_dir}/${name}/out.log
    python main_mbpo.py --automatic_entropy_tuning False \
                    --num_epoch 300 \
                    --env_name "Walker2d-v2" \
                    --epoch_length 1000 \
                    --gamma 0.99 \
                    --hidden_size 256 \
                    --init_exploration_steps 5000 \
                    --lr 0.0003 \
                    --max_path_length 1000 \
                    --max_train_repeat_per_step 5 \
                    --min_pool_size 1000 \
                    --model_retain_epochs 1 \
                    --model_train_freq 250 \
                    --model_type "pytorch" \
                    --num_train_repeat 20 \
                    --policy "Gaussian" \
                    --policy_train_batch_size 256 \
                    --pred_hidden_size 200 \
                    --real_ratio 0.05 \
                    --replay_size 1000000 \
                    --reward_size 1 \
                    --rollout_batch_size 100000 \
                    --rollout_max_epoch 100 \
                    --rollout_max_length 1 \
                    --rollout_min_epoch 20 \
                    --rollout_min_length 1 \
                    --target_update_interval 1 \
                    --tau 0.005 \
                    --train_every_n_steps 1 \
                    --use_decay True \
                    --wandb_mode online \
                    --log_dir $save_dir \
                    --seed $seed \
                    --tags $setting \
                    --save_name $name \
                    --project_name $project_name
done

# echo ${pids[*]}
# echo ${names[*]}
# for pid in ${pids[*]}; do
#     wait $pid
# done

date