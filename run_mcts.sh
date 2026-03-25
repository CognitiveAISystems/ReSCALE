#!/bin/bash

while [[ $# -gt 0 ]]; do
  case "$1" in
    --save_path)
      save_path="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --simulations)
      simulations="$2"
      shift 2
      ;;
    --length)
      length="$2"
      shift 2
      ;;
    --actions)
      actions="$2"
      shift 2
      ;;
    --ct2_dir)
      ct2_dir="$2"
      shift 2
      ;;
    --critic_model_path)
      critic_model_path="$2"
      shift 2
      ;;
    --n_gpus)
      n_gpus="$2"
      shift 2
      ;;
    --method)
      method="$2"
      shift 2
      ;;
    --env_name)
      env_name="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

: "${save_path:?--save_path is required}"
: "${seed:?--seed is required}"
: "${simulations:?--simulations is required}"
: "${length:?--length is required}"
: "${actions:?--actions is required}"
: "${ct2_dir:?--ct2_dir is required}"
: "${critic_model_path:?--critic_model_path is required}"
: "${n_gpus:?--n_gpus is required}"
: "${method:?--method is required}"
: "${env_name:?--env_name is required}"

is_test=True

dir_name="${seed}"_sim-"${simulations}"_len-"${length}"_act-"${actions}"
save_dir="${save_path}"/"${dir_name}"
work_dir="${save_path}"/_"${dir_name}"
if [ -d "${save_dir}" ]; then
        echo "Directory ${save_dir} exists."
        exit 0
fi

rm -rf ${work_dir} && mkdir -p ${work_dir}

num_start_nodes=$(( simulations <= actions ? simulations : actions ))
script_path="tsllm/offline_rl/run.py"

torchrun --nproc_per_node=$n_gpus --master-port 29503 ${script_path} \
    --ct2_dir ${ct2_dir} \
    --critic_model_path ${critic_model_path} \
    --tokenizer_path ${critic_model_path} \
    --save_dir ${work_dir} \
    --env_name ${env_name} \
    --rollout_method ${method} \
    --tree_max_length $length \
    --tree_max_actions $actions \
    --test $is_test \
    --final_action_strategy visits \
    --num_simulations $simulations \
    --sequential_halving_start_nodes $num_start_nodes \
    --non_root_child_selection_mode gumbel \
    --seed ${seed} \
    --save_mcts_trees False \
    2>&1 && mv ${work_dir} ${save_dir}
