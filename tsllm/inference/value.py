import gc
import math

import torch
from typing import Union, List
from tsllm.model import ValueHeadedLLM
from tsllm.model.modeling_actor_critic import AutoModelForCausalLMWithValueHead
from tsllm.llm.text_generation import llm_gen_ct2
from transformers import AutoTokenizer
import re
import numpy as np


@torch.inference_mode()
def value_fn(
    critic: ValueHeadedLLM, tokenizer: AutoTokenizer, input_str: Union[List[str], str], **value_config
):
    if isinstance(input_str, str):
        input_str = [input_str]

    num_sequence = len(input_str)
    success = False
    exception = None
    all_batch_sizes = value_config.get("batch_sizes", [num_sequence])
    batch_sizes = [batch_size for batch_size in all_batch_sizes if batch_size < num_sequence]
    if len(batch_sizes) < len(all_batch_sizes):
        batch_sizes.insert(0, all_batch_sizes[-len(batch_sizes) - 1])

    for batch_size in batch_sizes:
        values = []
        n_batches = math.ceil(num_sequence / batch_size)
        try:
            for batch_id in range(n_batches):
                batch_input_str = input_str[batch_id * batch_size:(batch_id + 1) * batch_size]
                indices2pick = torch.LongTensor(
                    [len(tokenizer.encode(txt)) - 1 for txt in batch_input_str]
                )

                inputs = tokenizer(batch_input_str, return_tensors="pt", padding=True).to(critic.device)
                if "token_type_ids" in inputs:
                    inputs.pop("token_type_ids")

                value = critic(**inputs).value.cpu()
                value = value.gather(1, indices2pick.unsqueeze_(1)).squeeze_(1).float().numpy()
                values.append(value)

            values = np.concatenate(values)
            success = True
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()

                print(f'An error occurred during value estimation with batch size {batch_size}:', e)
                exception = e
            else:
                raise e

    if not success:
        raise exception

    return values


@torch.inference_mode()
def value_fn_rlhf(
    critic: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    input_str: Union[List[str], str],
):
    if isinstance(input_str, list):
        indices2pick = torch.LongTensor(
            [len(tokenizer.encode(txt)) - 1 for txt in input_str]
        )
    else:
        indices2pick = torch.LongTensor([len(tokenizer.encode(input_str)) - 1])
    inputs = tokenizer(input_str, return_tensors="pt", padding=True).to(critic.device)
    value = critic(**inputs, return_dict=True).value.cpu()
    value = value.gather(1, indices2pick.unsqueeze_(1)).squeeze_(1).float().numpy()
    return value


@torch.inference_mode()
def seq_value_fn(critic_model, tokenizer, input_str):
    input_ids = tokenizer(input_str, return_tensors="pt").input_ids.to(
        critic_model.device
    )
    value = critic_model(input_ids, return_dict=True).value
    return value.cpu().float().numpy()
