from trl import GRPOConfig

def get_default_grpo_config(run_name: str, num_gpus: int = 1) -> GRPOConfig:
    return GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=1e-6,
        warmup_steps=50,
        num_train_epochs=1,
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=0.1,
        beta=0.01,
        max_prompt_length=1024,
        max_completion_length=1024,
        per_device_train_batch_size=8,
        num_generations=8,
        gradient_accumulation_steps=int(16 / num_gpus),
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_only_model=True,
        use_vllm=True,
        vllm_device=f"cuda:{num_gpus-1}",
        vllm_gpu_memory_utilization=0.7 if num_gpus > 1 else 0.3,
        logging_steps=1,
        log_on_each_node=False,
        report_to="wandb",
    )


