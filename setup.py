from setuptools import setup, find_packages

setup(
    name="verifiers",
    version="0.1.3",
    author="William Brown",
    author_email="williambrown97@gmail.com",
    description="Verifiers for reinforcement learning with LLMs",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch",
        "setuptools",
        "accelerate",
        "peft",
        "wandb",
        "rich",
        "duckduckgo-search",
        "liger-kernel>=0.5.2",
        "vllm>=0.7.3",
        "brave-search>=0.1.8",
        "trl @ git+https://github.com/huggingface/trl.git",
    ],
)
