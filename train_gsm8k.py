import verifiers as vf
from verifiers.tools import calculator
from verifiers.prompts import CALCULATOR_FEW_SHOT

print(vf.__version__)

model_name = "Qwen/Qwen2-0.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Initialize tool environment for GSM8K
vf_env = vf.MathEnv(
    dataset="gsm8k")
dataset = vf_env.get_dataset()
eval_dataset = vf_env.get_eval_dataset(n=100)
rubric = vf_env.get_rubric()

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
run_name = "multimodal_" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(
    run_name=run_name
)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
    #eval_dataset=eval_dataset,
)

try:
    trainer.train()
except Exception:
    pass

trainer.vllm_client.close_communicator() 