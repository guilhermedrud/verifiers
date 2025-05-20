from typing import List, Dict, Any, Tuple
from datasets import Dataset

from verifiers.server.vllm_client import VLLMClient
from trl.trainer.grpo_trainer import RewardFunc



from verifiers.envs.simple_env import SimpleEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import MultimodalRubric
from verifiers.prompts import SIMPLE_PROMPT, MATH_FEW_SHOT
from verifiers.utils import preprocess_dataset, preprocess_img_dataset

from abc import abstractmethod
import json
import random
from typing import List, Dict, Sequence, Any, Union

from datasets import Dataset

from ..imports import LLM, SamplingParams  # type: ignore
from verifiers.envs.environment import Environment

class MultimodalEnv(SimpleEnv):
    def __init__(self,
                 dataset: Dataset,
                 system_prompt: str = SIMPLE_PROMPT,    
                 few_shot: List[Dict[str, str]] = MATH_FEW_SHOT[0],
                 fields: List[str | Tuple[str, ...]] = ["reasoning", "answer"],
                 **kwargs):
        super().__init__(system_prompt=system_prompt, few_shot=few_shot, **kwargs)
        self.parser = XMLParser(fields=fields)
        self.dataset_name = dataset
        self.dataset = dataset
        self.eval_dataset = None
        self.rubric = MultimodalRubric()
    
    def get_dataset(self, **kwargs: Any) -> Dataset:
        return preprocess_img_dataset(self.dataset)
    
    def get_eval_dataset(self, n: int = -1, **kwargs: Any) -> Dataset | None:
        if self.eval_dataset is None:
            self.eval_dataset = self.dataset
        if n > 0:
            return self.eval_dataset.shuffle().select(range(n)) # type: ignore
        return self.eval_dataset 
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    

    def generate(self, prompts: List[List[Dict[str, Any]]],
                 llm,
                 n,
                 repetition_penalty,
                 temperature,
                 top_p,
                 top_k,
                 min_p,
                 max_tokens,
                 guided_decoding_regex,

                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:

        ## print('Prompts: ', prompts)
        states = [{
            "messages": [m],
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

        prompt_f = (f"<|im_start|>system\nYou're a helpful assistant<|im_end|>\n"
          "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
          "{}<|im_end|>\n"
          "<|im_start|>assistant\n")

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
