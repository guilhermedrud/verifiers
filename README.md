# Verifiers: Reinforcement Learning with LLMs in Verifiable Environments

This repository contains a set of tools for reinforcement learning with LLMs in verifiable environments.

For now, it supports the TRL implementation of the GRPO algorithm via a [fork](git@github.com:willccbb/trl.git), and requires [vLLM](https://github.com/vllm-project/vllm/tree/main) for inference.

## Installation

We recommend installing via `uv`:

```bash
uv add verifiers
```

Or, if you're old-school:
```bash
pip install verifiers
```

## Usage

```python
import verifiers as vf
from trl import GRPOTrainer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.DoubleCheckEnv()
trainer = GRPOTrainer(
    model=model_name,
    processing_class=tokenizer,
    env=vf_env,
    reward_funcs=vf_env.get_rubric(),
    args=vf.get_default_grpo_config(run_name="doublecheck", num_gpus=1),
    train_dataset=vf_env.get_dataset(),
)
trainer.train()
vf_env.eval(batch_size=32)
```
See `examples/doublecheck.py` for a complete example.


## Features
- [X] Environments: `SimpleEnv`, `MathEnv`, `DoubleCheckEnv`, `CodeEnv`
- [X] Dataset formatting
- [X] Rubrics for math correctness + response formatting
- [X] Defaults for GRPO, model, tokenizer, etc.

## Roadmap

There are a number of features we're planning to support in the near future:
- [ ] Multi-step code execution in `CodeEnv` 
- [ ] TextArena games
- [ ] LLM judges
- [ ] A range of other environments (suggestions welcome!)
- [ ] PPO
- [ ] Potential interoperability with other RL libraries (veRL, OpenRLHF, open-instruct, oat, etc.)

Community contributions are appreciated and encouraged!

## Citation

If you use this code in your research, please cite:

```bibtex
@article{brown2025verifiers,
  title={Verifiers: Reinforcement Learning with LLMs in Verifiable Environments},
  author={Brown, William},
  year={2025}
}
```
