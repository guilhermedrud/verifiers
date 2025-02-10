from typing import List
from trl.trainer.grpo_trainer import RewardFunc 

from verifiers.parsers import XMLParser


class MathRubric:
    def __init__(self):
        self.parser = XMLParser(fields=["reasoning", "answer"])
        def correctness_reward_func(completions, answer, **kwargs) -> List[float]:
            responses = [self.parser.parse(c[0]['content']).answer for c in completions]
            return [1.0 if r == a else 0.0 for r, a in zip(responses, answer)]

        def xml_reward_func(completions, **kwargs) -> List[float]:
            def count_xml(text: str) -> float:
                count = 0
                for tag in ["reasoning", "answer"]:
                    count += 1 - abs(text.count(f"<{tag}>") - 1)
                    count += 1 - abs(text.count(f"</{tag}>") - 1)
                return 0.1 * count 
            return [count_xml(c[-1]['content']) for c in completions]

        self.reward_funcs = [correctness_reward_func, xml_reward_func]

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs # type: ignore
    