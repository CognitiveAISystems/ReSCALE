from pathlib import Path
from typing import Dict, List, Optional
import torch.distributed as dist
from tsllm.argparse_utils import str2bool
from tsllm.distributed.utils import (
    print_rank_0,
    print_with_rank,
    init_distributed,
    gather_scalar,
)
from tsllm.envs import get_env_datasets, get_default_query_str_builder
from tsllm.inference.trajectory_collector import _mcts_rollout_v1, _mcts_gumbel
from tsllm.inference.value import value_fn
from tsllm.llm.ct2_utils import load_ct2_model
from tsllm.model import load_critic_model
from tsllm.llm.text_generation import llm_gen_ct2
from tsllm.mcts.tree import MCTS
from tsllm.inference.evaluation.vote_utils import RESULT, AGG_FN_MAP
from tsllm.envs.base_env import INVALID_ANS
from transformers import AutoTokenizer
import torch
from functools import partial
import json
import jsonlines
import time
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from argparse import ArgumentParser
import os
import importlib
import random

from tsllm.offline_rl.utils import get_batch_sizes


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


CHOSEN_AGGR_METHODS = [RESULT]


def judge_ans(
    problem_str: str,
    extracted_groundtruth: str,
    output_list: List[str],
    v_list: List[float],
    aggration_mode: str,
    extract_answer_fn,
    judge_correct_fn,
):
    ans_list = [extract_answer_fn(txt) for txt in output_list]
    valid_ans_list, valid_v_list = [], []
    for i, ans in enumerate(ans_list):
        if ans != INVALID_ANS:
            valid_ans_list.append(ans)
            valid_v_list.append(v_list[i])

    if len(valid_ans_list) == 0:
        return 0

    # score_normalization: this is only necessary for [-1, 1] values
    valid_v_list = np.array(valid_v_list, dtype=float)
    valid_v_list -= valid_v_list.min()
    valid_v_list /= valid_v_list.max() + 1e-3
    valid_v_list = valid_v_list.tolist()
    aggregated_ans = AGG_FN_MAP[aggration_mode](valid_ans_list, valid_v_list)

    return (
        1 if judge_correct_fn(problem_str, extracted_groundtruth, aggregated_ans) else 0
    )


def get_correct_proportion(
    problem_str: str,
    extracted_groundtruth: str,
    output_list: List[str],
    extract_answer_fn,
    judge_correct_fn,
):
    correct_list = [
        1.0
        if judge_correct_fn(problem_str, extracted_groundtruth, extract_answer_fn(txt))
        else 0.0
        for txt in output_list
    ]
    if len(correct_list) > 0:
        return np.mean(correct_list).item()
    else:
        return 0.0


def zero_critic():
    def _call(texts):
        if isinstance(texts, str):
            texts = [texts]

        assert isinstance(texts, (list, tuple))
        return np.zeros((len(texts),), dtype=np.float32)

    return _call


@dataclass
class SearchArgs:
    # temperature used for llm generation
    temperature: float = 1.0
    # temperature used for MCTS tree expansion
    action_distribution_temperature: float = 1.0
    use_mean_logprob: bool = True

    # MCTS aggregation number
    num_mcts_aggregation: int = 1

    # which tree search methods to use
    #  ["mcts.get_next_action", "mcts.gumbel"]
    # "mcts.get_next_action" is AlphaZero MCTS
    # "mcts.gumbel" is ReSCALE (Gumbel MCTS)
    rollout_method: str = None

    # Tree Search building configs
    max_length: int = 8
    max_action: int = 6

    # general mcts hyperparameters for MCTS
    # Tree basic configs
    pb_c_init: float = 10

    # MCTS-alpha hyperparamerters
    num_simulations: int = 10
    reset_total_tree: bool = False
    mcts_sample: bool = False
    clear_tree: bool = False
    clear_subtrees: bool = False
    final_action_strategy: str = None
    sequential_halving_start_nodes: int = 10
    non_root_child_selection_mode: str = 'ucb'
    max_new_tokens: int = 64
    seed: int = 7

    use_gumbel_noise_in_gumbel_mcts: bool = True
    use_gumbel_noise_in_alpha_mcts: bool = False

    max_generation_batch_size: int = 8
    max_critic_batch_size: int = 6

    save_mcts_trees: bool = True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ct2_dir", type=str, required=True)
    parser.add_argument("--critic_model_path", type=str, required=False)
    parser.add_argument("--zero_critic", type=str2bool, required=False)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--state_dict_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="gsm8k")
    parser.add_argument("--test", type=str2bool, default=True)
    parser.add_argument("--is_few_shot", type=str2bool, default=False)
    parser.add_argument("--rollout_method", type=str, default="mcts.rap")
    parser.add_argument("--tree_max_length", type=int, default=8)
    parser.add_argument("--tree_max_actions", type=int, default=6)
    parser.add_argument("--final_action_strategy", type=str, choices=['visits', 'expected_value', 'max_value'],
                        default="visits")
    parser.add_argument("--sequential_halving_start_nodes", type=int, default=5)
    parser.add_argument("--num_simulations", type=int, default=5)
    parser.add_argument("--clear_subtrees", type=str2bool, default=False)
    parser.add_argument("--non_root_child_selection_mode", type=str, choices=['ucb', 'gumbel'], default='ucb')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--action_distribution_temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use_mean_logprob", type=str2bool, default=True)
    parser.add_argument("--use_gumbel_noise_in_gumbel_mcts", type=str2bool, default=True)
    parser.add_argument("--use_gumbel_noise_in_alpha_mcts", type=str2bool, default=False)
    parser.add_argument("--max_generation_batch_size", type=int, default=8)
    parser.add_argument("--max_critic_batch_size", type=int, required=False)
    parser.add_argument("--save_mcts_trees", type=str2bool, default=True)
    config = parser.parse_args()

    args_list = [
        {
            "temperature": config.temperature,
            "action_distribution_temperature": config.action_distribution_temperature,
            "max_length": config.tree_max_length,
            "max_action": config.tree_max_actions,
            "use_mean_logprob": config.use_mean_logprob,
            "pb_c_init": 3,
            "num_simulations": config.num_simulations,
            "num_mcts_aggregation": 1,
            "rollout_method": config.rollout_method,
            "reset_total_tree": False,
            "mcts_sample": False,
            "clear_tree": True,
            "seed": config.seed,
            "final_action_strategy": config.final_action_strategy,
            "sequential_halving_start_nodes": config.sequential_halving_start_nodes,
            "clear_subtrees": config.clear_subtrees,
            "non_root_child_selection_mode": config.non_root_child_selection_mode,
            "max_new_tokens": config.max_new_tokens,
            "use_gumbel_noise_in_gumbel_mcts": config.use_gumbel_noise_in_gumbel_mcts,
            "use_gumbel_noise_in_alpha_mcts": config.use_gumbel_noise_in_alpha_mcts,
            "max_generation_batch_size": config.max_generation_batch_size,
            "max_critic_batch_size": config.max_critic_batch_size,
            "save_mcts_trees": config.save_mcts_trees,
        },
    ]

    task_module = importlib.import_module(f"tsllm.envs.{config.env_name}")
    extract_answer = task_module.extract_answer
    extract_groundtruth = task_module.extract_groundtruth
    judge_correct = task_module.judge_correct

    save_dir = Path(config.save_dir) / config.env_name

    local_rank, world_size = init_distributed()

    print_rank_0("ENV: {}, test set: {}".format(config.env_name, config.test))
    train_ds, test_ds = get_env_datasets(config.env_name)
    if not config.test:
        test_ds = train_ds

    device = torch.device(f"cuda:{local_rank}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    if config.zero_critic:
        policy_forward_value = zero_critic()
    else:
        critic = load_critic_model(
            config.critic_model_path, config.state_dict_path, device
        )
        max_critic_batch_size = config.max_critic_batch_size if config.max_critic_batch_size else config.tree_max_actions
        critic_batch_sizes = get_batch_sizes(max_critic_batch_size)
        policy_forward_value = partial(value_fn, critic, tokenizer, batch_sizes=critic_batch_sizes)

    ############ CONVERT MODEL to CT2 files ###################
    ct2_generator, ct2_sp = load_ct2_model(
        config.ct2_dir, device="cuda", device_index=local_rank, compute_type="bfloat16"
    )

    def prompt_fn(problem_input: str):
        return get_default_query_str_builder(config.env_name)(
            problem_input, is_few_shot=config.is_few_shot
        )

    def mcts_multi_search(
        args: "SearchArgs", problem, no_terminal_reward=True, tree_path=None
    ):
        env = task_module.Env(
            config={
                "max_actions": args.max_action,
                "max_length": args.max_length,
                "generation_config": {
                    "max_new_tokens": 64,
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": 1.0,
                    "top_k": 100,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "use_cache": True,
                    "use_mean_logprob": args.use_mean_logprob,
                    "generation_batch_sizes": get_batch_sizes(args.max_generation_batch_size),
                },
            },
            math_problems=[
                {
                    "question": problem[task_module.QUESTION_KEY],
                    "answer": extract_groundtruth(problem["answer"]),
                }
            ],
            llm_gen_fn=partial(llm_gen_ct2, ct2_generator, tokenizer),
            tokenizer=tokenizer,
            action_distribution_temperature=args.action_distribution_temperature,
        )
        # llm_gen_fn=partial(llm_gen_with_logp_v1, model, tokenizer),
        cfg = {
            "num_simulations": args.num_simulations,
            "pb_c_base": 19652,
            "pb_c_init": args.pb_c_init,
            "root_dirichlet_alpha": 0.3,
            "root_noise_weight": 0.25,
            "no_terminal_reward": no_terminal_reward,
            "final_action_strategy": args.final_action_strategy,
            "sequential_halving_start_nodes": args.sequential_halving_start_nodes,
            "non_root_child_selection_mode": args.non_root_child_selection_mode,
            "use_gumbel_noise_in_gumbel_mcts": args.use_gumbel_noise_in_gumbel_mcts,
            "use_gumbel_noise_in_alpha_mcts": args.use_gumbel_noise_in_alpha_mcts,
        }
        if tree_path and tree_path.exists():
            mcts = MCTS.from_json(cfg, tree_path, reset_visit_info=True)
        else:
            mcts = MCTS(cfg=cfg)

        if args.rollout_method == "mcts.get_next_action":
            output_list, tree = _mcts_rollout_v1(
                mcts,
                env,
                policy_forward_value,
                args.num_mcts_aggregation,
                args.reset_total_tree,
                sample=args.mcts_sample,
                clear_total_tree=args.clear_tree,
            )
            prompt = prompt_fn(problem[task_module.QUESTION_KEY])
            texts = [o["text"] for o in output_list]
            if len(texts) > 0:
                value_list = policy_forward_value(
                    # add a .strip() in case mistakes happens when copy this line to other place
                    [prompt + txt.strip() + task_module.SEP for txt in texts]
                ).tolist()
            else:
                value_list = []
            for o, v in zip(output_list, value_list):
                o["value"] = v

        elif args.rollout_method == "mcts.gumbel":
            output_list, tree = _mcts_gumbel(
                mcts,
                env,
                policy_forward_value,
                args.num_mcts_aggregation,
                args.reset_total_tree,
                sample=args.mcts_sample,
                clear_total_tree=args.clear_tree,
                clear_subtrees=args.clear_subtrees,
            )
            prompt = prompt_fn(problem[task_module.QUESTION_KEY])
            texts = [o["text"] for o in output_list]
            if len(texts) > 0:
                value_list = policy_forward_value(
                    # add a .strip() in case mistakes happens when copy this line to other place
                    [prompt + txt.strip() + task_module.SEP for txt in texts]
                ).tolist()
            else:
                value_list = []
            for o, v in zip(output_list, value_list):
                o["value"] = v
        else:
            raise ValueError("Unknow rollout method: {}".format(args.rollout_method))

        texts = [o["text"] for o in output_list]
        value_list = [o["value"] for o in output_list]

        extracted_groundtruth = extract_groundtruth(problem["answer"])
        judge_results = {
            f"{k}@{args.num_mcts_aggregation}": judge_ans(
                problem[task_module.QUESTION_KEY],
                extracted_groundtruth,
                texts,
                value_list,
                k,
                extract_answer,
                judge_correct,
            )
            for k in CHOSEN_AGGR_METHODS
        }
        judge_results["c%"] = get_correct_proportion(
            problem[task_module.QUESTION_KEY],
            extracted_groundtruth,
            texts,
            extract_answer,
            judge_correct,
        )

        tree['groundtruth'] = extracted_groundtruth
        tree['correct'] = bool(judge_results[f'{RESULT}@{args.num_mcts_aggregation}'])
        num_token = output_list[-1]["num_generated_token"]
        judge_results["#token"] = num_token
        return mcts, judge_results, output_list, tree

    def test_problem(
        args,
        idx,
        problem_inst,
        mcts_no_term_writer,
        mcts_no_term_tree_writer,
    ):
        results = {}

        def save_fn(writer, output, result: Dict):
            if writer is not None:
                obj = {
                    "i": idx,
                    "question": problem_inst[task_module.QUESTION_KEY],
                    "groundtruth": problem_inst["answer"],
                    "output": output,
                    "result": result,
                }
                writer.write(obj)

        def save_tree_fn(writer, tree):
            if writer is not None:
                tree['idx'] = idx
                writer.write(tree)

        mcts, r_no_terminal, no_terminal_episodes, tree = mcts_multi_search(args, problem_inst, True)
        save_fn(mcts_no_term_writer, no_terminal_episodes, r_no_terminal)
        save_tree_fn(mcts_no_term_tree_writer, tree)
        results["w/o-terminal"] = r_no_terminal

        return results

    def _result_str(results, cnt, join_str="\n"):
        res = ""
        for k, v in results.items():
            if isinstance(v, int):
                res += f"{k}: {v/cnt:.2%}"
            elif isinstance(v, dict):
                res += f"{k}: "
                res += ", ".join(
                    [
                        (
                            f"{sub_k}: {sub_v/cnt:.2f}"
                            if sub_k == "#token"
                            else f"{sub_k}: {sub_v/cnt:.2%}"
                        )
                        for sub_k, sub_v in v.items()
                    ]
                )
            else:
                raise ValueError
            res += join_str
        res += f"cnt: {cnt}"
        return res

    for i_arg, cur_args in enumerate(args_list):
        args = SearchArgs(**cur_args)
        seed = args.seed
        setup_seed(seed)
        writer_dir = save_dir / (f"args{i_arg}_seed{seed}/")

        if local_rank == 0:
            print("Search args: {}, SEED={}".format(args, seed))
            if not writer_dir.exists():
                writer_dir.mkdir(parents=True)
            json.dump(cur_args, open(writer_dir / "args.json", "w"))

        mcts_no_term_save_path = writer_dir / "output"
        if local_rank == 0 and not mcts_no_term_save_path.exists():
            mcts_no_term_save_path.mkdir(parents=True)
        dist.barrier()
        mcts_no_term_writer = jsonlines.open(
            mcts_no_term_save_path / f"{local_rank}.jsonl", "a"
        )
        if args.save_mcts_trees:
            mcts_no_term_tree_writer = jsonlines.open(
                mcts_no_term_save_path / f"{local_rank}_tree.jsonl", "a"
            )
        else:
            mcts_no_term_tree_writer = None

        cnt = 0
        correct_cnt_dict = dict()
        t0 = time.time()
        for i in (pbar := tqdm(range(len(test_ds)), disable=(local_rank != 0))):
            if i % world_size == local_rank:
                results = test_problem(
                    args,
                    i,
                    test_ds[i],
                    mcts_no_term_writer,
                    mcts_no_term_tree_writer,
                )
                for k, v in results.items():
                    if isinstance(v, int):
                        if k not in correct_cnt_dict:
                            correct_cnt_dict[k] = 0
                        correct_cnt_dict[k] += v
                    elif isinstance(v, dict):
                        if k not in correct_cnt_dict:
                            correct_cnt_dict[k] = dict()
                        for sub_k, sub_v in v.items():
                            if sub_k not in correct_cnt_dict[k]:
                                correct_cnt_dict[k][sub_k] = 0

                            correct_cnt_dict[k][sub_k] += sub_v
                cnt += 1
                results_strs = _result_str(correct_cnt_dict, cnt, join_str="; ")
                pbar.set_description(results_strs)

        print_with_rank(results_strs)

        cnt_list = gather_scalar(cnt, local_rank, world_size)

        gathered_results = {}
        for k, v in correct_cnt_dict.items():
            if isinstance(v, int):
                gathered_list = gather_scalar(int(v), local_rank, world_size)
                if local_rank == 0:
                    gathered_results[k] = sum(gathered_list)
            elif isinstance(v, dict):
                gathered_results[k] = {}
                for sub_k, sub_v in v.items():
                    gathered_list = gather_scalar(float(sub_v), local_rank, world_size)
                    if local_rank == 0:
                        gathered_results[k][sub_k] = sum(gathered_list)
            else:
                raise ValueError

        if local_rank == 0:
            total_cnt = sum(cnt_list)
            t1 = time.time()
            total_results_strs = _result_str(gathered_results, total_cnt)

            print(cur_args)
            print("TOTAL RESULTS:\n", total_results_strs)
            print("Time: {}".format(t1 - t0))
