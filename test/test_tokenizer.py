"""Tests for BPE tokenizer."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from tokenizer.bpe import BPETokenizer

SAMPLE_CORPUS = """
the entity violated sebi regulations by failing to disclose material information
the data controller obtained consent from all data subjects prior to processing
insider trading refers to buying or selling securities with unpublished price sensitive information
gdpr requires data minimisation purpose limitation and storage limitation principles
the broker must segregate client funds and maintain proper kyc documentation
compliance with sebi lodr regulations requires timely disclosure of material events
personal data must be processed lawfully fairly and in a transparent manner
the company maintained proper records of all transactions as required by regulations
"""

@pytest.fixture(scope="module")
def trained_tokenizer():
    tok = BPETokenizer(vocab_size=500)
    tok.train(SAMPLE_CORPUS, verbose=False)
    return tok

def test_vocab_size(trained_tokenizer):
    assert len(trained_tokenizer) <= 500

def test_special_tokens_present(trained_tokenizer):
    for name, idx in BPETokenizer.SPECIAL_TOKENS.items():
        assert name in trained_tokenizer.vocab

def test_encode_decode_roundtrip(trained_tokenizer):
    text  = "sebi regulations require disclosure"
    ids   = trained_tokenizer.encode(text, add_special_tokens=False)
    recon = trained_tokenizer.decode(ids)
    for word in ["sebi", "regulations"]:
        assert word in recon

def test_encode_adds_special_tokens(trained_tokenizer):
    ids = trained_tokenizer.encode("test text", add_special_tokens=True)
    assert ids[0]  == trained_tokenizer.bos_id
    assert ids[-1] == trained_tokenizer.eos_id

def test_max_length_truncation(trained_tokenizer):
    ids = trained_tokenizer.encode("sebi " * 200, max_length=32, add_special_tokens=True)
    assert len(ids) == 32

def test_padding(trained_tokenizer):
    ids = trained_tokenizer.encode("short", max_length=64, padding=True, add_special_tokens=True)
    assert len(ids) == 64

def test_save_load(trained_tokenizer, tmp_path):
    path   = str(tmp_path / "tokenizer.json")
    trained_tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    text   = "sebi violation"
    assert trained_tokenizer.encode(text) == loaded.encode(text)
