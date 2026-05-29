from setuptools import setup, find_packages

setup(
    name="legalmind",
    version="1.0.0",
    description="Legal Compliance Violation Classifier — End-to-end LLM pipeline from scratch",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().strip().splitlines(),
    entry_points={
        "console_scripts": [
            "lm-tokenize=scripts.train_tokenizer:main",
            "lm-pretrain=scripts.run_pretrain:main",
            "lm-finetune=scripts.run_finetune:main",
            "lm-serve=scripts.serve:main",
        ],
    },
)
