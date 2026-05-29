"""
Central configuration for LegalMind.
All hyperparameters live here — modify this file to experiment.

Default model: ~14.5M parameters
  - 6 transformer layers, d_model=512
  - GQA: 8 query heads / 2 kv heads (4:1 ratio)
  - Context window: 256 tokens
  - BPE vocab: 8000 tokens
"""
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """BPE tokenizer training settings."""
    vocab_size: int = 8000
    min_pair_freq: int = 2       # Minimum frequency to keep a merge rule
    save_path: str = "tokenizer.json"


@dataclass
class ModelConfig:
    """GPT architecture — all dimensions flow from d_model."""
    vocab_size: int = 8000
    context_length: int = 256    # Max tokens the model can attend to
    n_layers: int = 6
    n_heads: int = 8             # Query attention heads
    n_kv_heads: int = 2          # Key/Value heads (GQA). Must divide n_heads evenly.
    d_model: int = 512           # Embedding / hidden dimension
    d_ff: int = 1024             # Feed-forward hidden size (typically 2-4x d_model)
    dropout: float = 0.1
    bias: bool = False           # No bias in linear layers (like GPT-NeoX / Llama)
    num_classes: int = 2         # 0=compliant, 1=violation


@dataclass
class PretrainConfig:
    """Pretraining loop settings (next-token prediction on legal corpus)."""
    batch_size: int = 8          # Keep small for 8 GB RAM
    seq_len: int = 256
    learning_rate: float = 3e-4
    min_lr: float = 3e-5         # Cosine LR decay floor
    max_steps: int = 20_000      # Total gradient update steps
    warmup_steps: int = 200
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True   # Trade compute for memory
    eval_interval: int = 500
    save_interval: int = 2000
    checkpoint_dir: str = "checkpoints/pretrain"
    log_interval: int = 50


@dataclass
class FinetuneConfig:
    """Fine-tuning loop settings (classification head)."""
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    warmup_steps: int = 50
    grad_clip: float = 1.0
    early_stopping_patience: int = 3
    use_class_weights: bool = True   # Handle label imbalance
    checkpoint_dir: str = "checkpoints/finetune"
    pretrain_ckpt: str = "checkpoints/pretrain/best.pt"


@dataclass
class APIConfig:
    """FastAPI inference server settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    use_bf16: bool = True              # bf16 for faster CPU inference
    model_path: str = "checkpoints/finetune/best.pt"
    tokenizer_path: str = "tokenizer.json"
    drift_window: int = 500            # Samples for drift reference window
    drift_kl_threshold: float = 0.1   # KL divergence alert threshold


# Ready-to-import defaults
TOKENIZER_CFG = TokenizerConfig()
MODEL_CFG     = ModelConfig()
PRETRAIN_CFG  = PretrainConfig()
FINETUNE_CFG  = FinetuneConfig()
API_CFG       = APIConfig()
