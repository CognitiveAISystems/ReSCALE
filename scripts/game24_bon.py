import argparse
import json
from functools import partial

import jsonlines


from transformers import AutoTokenizer
import ctranslate2
import numpy as np
from tqdm import tqdm

from tsllm.inference.value import value_fn
from tsllm.model import load_critic_model
from tsllm.envs.game24.env import extract_answer, judge_correct, PROBLEM_FORMAT_STR


def check_answer(answer_str, problem_str):
    answer = extract_answer(answer_str)
    if answer is None:
        return False

    return judge_correct(problem_str, None, answer)


def generate_answers(dataset_path, tokenizer, ct2_path, n):
    ds = []
    with jsonlines.open(dataset_path) as reader:
        for obj in reader:
            ds.append(obj)

    ct2_generator = ctranslate2.Generator(ct2_path, device="cuda", device_index=0, compute_type="bfloat16")

    completions = []
    results = []
    n_tokens = []
    questions = []
    for record in tqdm(ds):
        completions.append([])
        results.append([])
        n_tokens.append([])
        question = record["question"]
        questions.append(question)
        prompt = PROBLEM_FORMAT_STR.format(question=question)
        prompt_tokens = tokenizer.tokenize(prompt)
        generations = ct2_generator.generate_batch(
            [prompt_tokens],
            sampling_temperature=args.temperature,
            sampling_topp=1,
            sampling_topk=args.topk,
            max_length=2048,
            # return_logits_vocab=True,
            include_prompt_in_result=False,
            end_token=[tokenizer.eos_token_id],
            num_hypotheses=n,
            return_end_token=True,
        )
        assert len(generations) == 1
        for seq in generations[0].sequences:
            completion_token_ids = tokenizer.convert_tokens_to_ids(seq)
            completion = tokenizer.decode(completion_token_ids, skip_special_tokens=True, )
            is_correct = check_answer(completion, question)
            completions[-1].append(completion)
            results[-1].append(is_correct)
            n_tokens[-1].append(len(completion_token_ids))

    return questions, completions, results, n_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ct2_path", type=str)
    parser.add_argument("--critic_model_path", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.critic_model_path)
    questions, completions, results, n_tokens = generate_answers(args.dataset_path, tokenizer, args.ct2_path, args.n)
    critic = load_critic_model(args.critic_model_path, None, args.device)
    policy_forward_value = partial(value_fn, critic, tokenizer)

    is_correct = []
    all_results = []
    for question, completion, result, token in zip(questions, completions, results, n_tokens):
        value_completion = [f'{question}\n{c}\n' for c in completion]
        values = policy_forward_value(value_completion)
        max_index = np.argmax(values)
        is_correct.append(result[max_index])
        answers = []
        for i, (c, r, v, t) in enumerate(zip(completion, result, values, token)):
            record = {'answer': c, 'is_correct': r, 'value': float(v), 'tokens': t, 'is_max_value': bool(max_index == i)}
            answers.append(record)

        all_results.append(dict(question=question, answers=answers))

    is_correct = np.array(is_correct)
    print(f'acc={is_correct.mean()}', sum(sum(t) for t in n_tokens) / len(n_tokens))
    with open(args.output, 'w') as f:
        json.dump(all_results, f)
