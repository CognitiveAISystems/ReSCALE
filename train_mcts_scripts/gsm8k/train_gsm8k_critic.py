import argparse

from tsllm.rl.trainer.mcts_trainer_traj_ct2_value import AccelerateMCTSTrainer
from tsllm.rl.config import RLConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='Tree_Search')
    parser.add_argument('--tracker', type=str, choices=['tensorboard', 'wandb'], required=False)
    parser.add_argument('--wandb_project_name', type=str, required=False)
    parser.add_argument('--wandb_entity_name', type=str, required=False)
    parser.add_argument('--wandb_group_name', type=str, required=False)
    parser.add_argument('--micro_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = {
        "model": {
            "model_path": "meta-llama/Llama-2-7b-hf",
        },
        "tokenizer": {
            "tokenizer_path": "meta-llama/Llama-2-7b-hf",
            "padding_side": "right",
        },
        "optimizer": {
            "name": "adamw",
            "kwargs": dict(lr=2.0e-5, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0),
        },
        "scheduler": {"name": "cosine_warmup", "kwargs": dict(warmup_ratio=0.03)},
        "train": {
            "pre_onpolicy_datapath": "../../tsllm/offline_rl/gsm8k_data/processed/gsm8k_train_cot_sample_sft_k100_merged_dedup_sample17x3.jsonl",
            "pre_onpolicy_datapath_train_test": "../../tsllm/offline_rl/gsm8k_data/processed/gsm8k_train_cot_sample_offline_sft_k100_ep3_dedup_sample17_train_test_sample_3.jsonl",
            "env_name": "gsm8k",
            "epochs": 3,  # this is the epoch for the whole sampling/training process
            "train_epoch": 1,  # this is the epoch for training process after each sampling
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "seq_length": 1024,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "value_loss_coef": 1.0,
            "eval_interval": 1,
            "checkpoint_interval": 1,
            "checkpoint_dir": args.checkpoint_dir,
            "save_optimizer": False,
            "project_name": args.wandb_project_name,
            "tracker": args.tracker,
            "entity_name": args.wandb_entity_name,
            "group_name": args.wandb_group_name,
            "tags": ["gsm8k_sft"],
            "logging_dir": "logs/",
            "onpolicy_per_problem_max_size": 1000,
        },
        "mcts": {},
        "env": {},
    }

    config = RLConfig.from_dict(config)
    trainer = AccelerateMCTSTrainer(config)

    trainer.learn()
