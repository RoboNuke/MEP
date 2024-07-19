#TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 python data_analysis/workspace_vis.py
# create folder for this round of experiments
exp_set_name="DNH_RGB"
number_of_seeds=10

num_envs=256
env_id="StackCube-v1"
update_epochs=8
num_minibatches=8
total_timesteps=50000000
eval_freq=20

date=$(date +"%Y-%m-%d_%H:%M")
filepath="${exp_set_name}_${env_id}_${date}"
#echo $filepath

for i in $(seq 1 $number_of_seeds);
do
    printf "\n\n\n\nStarting baseline exp ${i}\n\n\n\n"
    #exp_name = "pickcube_state_baseline_" + $i
    exp_name="${filepath}/baseline/state_${i}"
    python -m trainers.PPO_rgb --env_id=$env_id \
    --num_envs=$num_envs --update_epochs=$update_epochs \
    --num_minibatches=$num_minibatches --seed=$i\
    --total_timesteps=$total_timesteps --eval_freq=$eval_freq \
    --exp_name=$exp_name --no-with-MEP-wrapper
done

for i in $(seq 1 $number_of_seeds);
do
    printf "\n\n\n\nStarting MEP exp ${i}\n\n\n\n"
    exp_name="${filepath}/MEP/state_${i}"
    python -m trainers.PPO_rgb --env_id=$env_id \
    --num_envs=$num_envs --update_epochs=$update_epochs \
    --num_minibatches=$num_minibatches --seed=$i\
    --total_timesteps=$total_timesteps --eval_freq=$eval_freq \
    --exp_name=$exp_name --with-MEP-wrapper
done


# store data
python -m data_analysis.tb_extraction $filepath RGB $number_of_seeds "runs/${filepath}/"

# store plots
mkdir -p "runs/${filepath}/plots/"
python -m data_analysis.pd_graphing "runs/${filepath}/" $env_id "World Camera"




# python sac_test.py --env_id="PushCube-v1" \
#  --num_envs=32 --utd=0.5 --buffer_size=200_000 \
#  --total_timesteps=200_000 --eval_freq=50_000

#python ppo.py --env_id="PushCube-v1" \
#  --num_envs=2048 --update_epochs=8 --num_minibatches=32 \
#  --total_timesteps=2_000_000 --eval_freq=10 --num-steps=20

#python PPO_training.py --env_id="PushCube-v1" \
#  --num_envs=256 --update_epochs=8 --num_minibatches=8 \
#  --total_timesteps=1_000_000 --eval_freq=10 --num-steps=20

#python PPO_training.py --env_id="PickCube-v1" \
#  --num_envs=300 --update_epochs=8 --num_minibatches=8 \
#  --total_timesteps=10_000_000 --eval_freq=10