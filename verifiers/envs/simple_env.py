from abc import abstractmethod
import json
import random
from typing import List, Dict, Sequence, Any, Union
from verifiers.server.vllm_client import VLLMClient

from datasets import Dataset

from ..imports import LLM, SamplingParams  # type: ignore
from verifiers.envs.environment import Environment


class SimpleEnv(Environment):
    def __init__(self,
                 system_prompt: str = "",
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.few_shot = few_shot
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1
        }
        self.sampling_args.update(sampling_args)

    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    def format_prompt(self, prompt: str, fewshot_prob: float = 1.0) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.few_shot and random.random() < fewshot_prob:
            messages.extend(self.few_shot)
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm: LLM,

                 n,
                 repetition_penalty,
                 temperature,
                 top_p,
                 top_k,
                 min_p,
                 max_tokens,
                 guided_decoding_regex,

                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:

        states = [{
            "messages": m,
            "prompt_ids": [],
            "completion_ids": [],
            "completion_mask": []
        } for m in prompts]

        # get completions
        # completions = llm.chat(prompts, sampling_params=custom_sp, use_tqdm=False) # type: ignore
        # for i, completion in enumerate(completions):
        #     states[i]["messages"].append({"role": "assistant", "content": completion.outputs[0].text})
        #     states[i]["prompt_ids"] = list(completion.prompt_token_ids) # type: ignore
        #     states[i]["completion_ids"] = list(completion.outputs[0].token_ids)
        #     states[i]["completion_mask"] = [1] * len(states[i]["completion_ids"])

        completions = llm.chat(prompts, n=n, repetition_penalty=repetition_penalty, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, max_tokens=max_tokens, guided_decoding_regex=guided_decoding_regex) # type: ignore
        for i, completion in enumerate(completions):
            states[i]["messages"].append({"role": "assistant", "content": completion['outputs']['text']})
            states[i]["prompt_ids"] = list(completion['prompt_token_ids']) # type: ignore
            states[i]["completion_ids"] = list(completion['outputs']['token_ids'])
            states[i]["completion_mask"] = [1] * len(states[i]["completion_ids"])

        output = {
            "ids": [states[i]["completion_ids"] for i in range(len(states))],
            "messages": [states[i]["messages"][-1:] for i in range(len(states))],
            "mask": [states[i]["completion_mask"] for i in range(len(states))]
        }
        return output