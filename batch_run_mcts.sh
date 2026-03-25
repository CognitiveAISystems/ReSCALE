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
    --parameters_path)
      parameters_path="$2"
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
: "${ct2_dir:?--ct2_dir is required}"
: "${critic_model_path:?--critic_model_path is required}"
: "${n_gpus:?--n_gpus is required}"
: "${method:?--method is required}"
: "${parameters_path:?--parameters_path is required}"
: "${env_name:?--env_name is required}"

[[ -f "$parameters_path" ]] || { echo "File not found: $parameters_path"; exit 1; }

{
  read -r
  while IFS=$'\t' read -r simulations   actions length tokens accuracy; do
    bash run_mcts.sh --save_path ${save_path} --seed ${seed} --simulations ${simulations} --length ${length} \
    --actions ${actions} --ct2_dir ${ct2_dir} --critic_model_path ${critic_model_path} --n_gpus ${n_gpus} \
    --method ${method} --env_name ${env_name}
  done
} < <(grep -v '^[[:space:]]*$' "$parameters_path")

