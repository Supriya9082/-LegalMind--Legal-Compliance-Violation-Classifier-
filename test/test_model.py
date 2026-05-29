"""Tests for GPT model."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest, torch
from model.gpt import LegalMindGPT

TINY = dict(vocab_size=100, context_length=32, n_layers=2, n_heads=4,
            n_kv_heads=2, d_model=64, d_ff=128, dropout=0.0, bias=False)

@pytest.fixture
def pretrain_model(): return LegalMindGPT(**TINY, num_classes=0)

@pytest.fixture
def finetune_model(): return LegalMindGPT(**TINY, num_classes=2)

def test_pretrain_forward_shape(pretrain_model):
    ids = torch.randint(0, 100, (2, 16))
    _, logits = pretrain_model(ids, mode="pretrain")
    assert logits.shape == (2, 16, 100)

def test_pretrain_loss(pretrain_model):
    ids = torch.randint(0, 100, (2, 16))
    loss, _ = pretrain_model(ids, labels=ids, mode="pretrain")
    assert loss.item() > 0

def test_finetune_forward(finetune_model):
    ids  = torch.randint(0, 100, (4, 16))
    mask = torch.ones(4, 16, dtype=torch.long)
    lbls = torch.randint(0, 2, (4,))
    loss, logits = finetune_model(ids, attention_mask=mask, labels=lbls, mode="finetune")
    assert logits.shape == (4, 2)
    assert loss.item() > 0

def test_weight_tying(pretrain_model):
    assert pretrain_model.lm_head.weight is pretrain_model.tok_emb.weight

def test_freeze_unfreeze(finetune_model):
    finetune_model.freeze_backbone()
    for n, p in finetune_model.named_parameters():
        if "cls_head" not in n:
            assert not p.requires_grad
    finetune_model.unfreeze_all()

def test_generate(pretrain_model):
    pretrain_model.eval()
    prompt = torch.randint(0, 100, (1, 8))
    out = pretrain_model.generate(prompt, max_new_tokens=10)
    assert out.shape[1] == 18

def test_gradient_checkpointing(pretrain_model):
    pretrain_model.train()
    ids = torch.randint(0, 100, (2, 16))
    loss, _ = pretrain_model(ids, labels=ids, use_checkpoint=True, mode="pretrain")
    loss.backward()
    assert loss.item() > 0
