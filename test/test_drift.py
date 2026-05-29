"""Tests for drift monitor."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from inference.drift import DriftMonitor, kl_divergence

def test_warmup_phase():
    mon = DriftMonitor(window=100, threshold=0.1, vocab_size=500, warmup=50)
    for _ in range(30):
        mon.add(list(range(20)))
    assert mon.check()["drift_detected"] is False

def test_no_drift_similar_data():
    mon = DriftMonitor(window=100, threshold=0.1, vocab_size=500, warmup=20)
    import random
    for _ in range(80):
        mon.add([random.randint(0, 10) for _ in range(20)])
    assert mon.check()["drift_detected"] is False

def test_drift_detected():
    mon = DriftMonitor(window=200, threshold=0.05, vocab_size=500, warmup=30)
    for _ in range(30):
        mon.add(list(range(10)))
    for _ in range(50):
        mon.add(list(range(490, 500)))
    result = mon.check()
    assert result["drift_detected"] is True

def test_kl_identical():
    p = {i: 0.1 for i in range(10)}
    assert kl_divergence(p, p, vocab_size=100) < 1e-6

def test_reset():
    mon = DriftMonitor(window=100, threshold=0.01, vocab_size=500, warmup=10)
    for _ in range(10): mon.add(list(range(5)))
    for _ in range(50): mon.add(list(range(490, 500)))
    mon.check()
    mon.reset_reference()
    assert mon.alert_count == 0
