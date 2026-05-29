"""
FastAPI inference server for LegalMind.

Endpoints:
  POST /predict        — single text classification
  POST /predict/batch  — batch classification
  GET  /health         — liveness probe
  GET  /drift          — drift monitor status
  POST /drift/reset    — reset drift reference window
  GET  /metrics        — prediction statistics

Run:
  uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Global state (loaded once at startup) ────────────────────────────────────
_predictor = None
_monitor   = None
_stats     = {"total": 0, "violations": 0, "errors": 0, "latencies": []}


# ── Lifespan (replaces deprecated @app.on_event) ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + drift monitor on startup; clean up on shutdown."""
    global _predictor, _monitor

    from config import API_CFG, MODEL_CFG
    from inference.predictor import CompliancePredictor
    from inference.drift import DriftMonitor

    logger.info("[Server] Loading model...")
    try:
        _predictor = CompliancePredictor(
            model_path     = API_CFG.model_path,
            tokenizer_path = API_CFG.tokenizer_path,
            use_bf16       = API_CFG.use_bf16,
        )
        _monitor = DriftMonitor(
            window     = API_CFG.drift_window,
            threshold  = API_CFG.drift_kl_threshold,
            vocab_size = MODEL_CFG.vocab_size,
        )
        logger.info("[Server] Ready ✓")
    except Exception as e:
        logger.error(f"[Server] Failed to load model: {e}")
        logger.warning("[Server] Starting without model (health endpoint still active)")

    yield   # ← server runs here

    logger.info("[Server] Shutting down")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "LegalMind API",
    description = "Legal compliance violation classifier — 15M parameter GPT",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=10_000,
                      example="The broker failed to disclose material information to clients.")

class PredictResponse(BaseModel):
    label:       str
    label_id:    int
    confidence:  float
    scores:      dict
    latency_ms:  float
    drift_alert: bool = False

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64)

class BatchPredictResponse(BaseModel):
    results:     List[dict]
    total_ms:    float
    drift_alert: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness probe — always returns 200 if server is running."""
    return {
        "status":       "ok",
        "model_loaded": _predictor is not None,
        "uptime_reqs":  _stats["total"],
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Classify a single legal text snippet as compliant or violation.

    Returns label, confidence score, and per-class probabilities.
    """
    if _predictor is None:
        raise HTTPException(503, "Model not loaded. Check server logs.")

    try:
        result = _predictor.predict(req.text)
    except Exception as e:
        _stats["errors"] += 1
        logger.error(f"[/predict] Error: {e}")
        raise HTTPException(500, f"Inference error: {str(e)}")

    # ── Update drift monitor ──────────────────────────────────────
    token_ids = _predictor.tokenizer.encode(req.text, add_special_tokens=False)
    _monitor.add(token_ids)
    drift_info = _monitor.check()

    # ── Update stats ──────────────────────────────────────────────
    _stats["total"] += 1
    _stats["latencies"].append(result["latency_ms"])
    if len(_stats["latencies"]) > 1000:
        _stats["latencies"] = _stats["latencies"][-1000:]
    if result["label_id"] == 1:
        _stats["violations"] += 1

    return PredictResponse(
        **result,
        drift_alert = drift_info["drift_detected"],
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest):
    """Classify a batch of texts. More efficient than N single requests."""
    if _predictor is None:
        raise HTTPException(503, "Model not loaded.")

    t0 = time.perf_counter()
    results = []
    drift_alert = False

    for text in req.texts:
        try:
            result = _predictor.predict(text)
            token_ids = _predictor.tokenizer.encode(text, add_special_tokens=False)
            _monitor.add(token_ids)
            drift_info = _monitor.check()
            if drift_info["drift_detected"]:
                drift_alert = True
            results.append(result)
            _stats["total"] += 1
        except Exception as e:
            _stats["errors"] += 1
            results.append({"error": str(e), "text": text[:50]})

    total_ms = (time.perf_counter() - t0) * 1000
    return BatchPredictResponse(results=results, total_ms=round(total_ms, 2),
                                drift_alert=drift_alert)


@app.get("/drift")
async def drift_status():
    """Current drift monitor status and KL divergence statistics."""
    if _monitor is None:
        return {"status": "monitor not initialised"}
    return {**_monitor.check(), **_monitor.stats}


@app.post("/drift/reset")
async def drift_reset():
    """Reset drift reference window (call after model update / retraining)."""
    if _monitor is None:
        raise HTTPException(503, "Monitor not initialised.")
    _monitor.reset_reference()
    return {"status": "reference reset", "samples_seen": _monitor._n_seen}


@app.get("/metrics")
async def server_metrics():
    """Aggregate prediction statistics."""
    lats = _stats["latencies"]
    p50  = sorted(lats)[int(len(lats) * 0.50)] if lats else 0
    p95  = sorted(lats)[int(len(lats) * 0.95)] if lats else 0
    p99  = sorted(lats)[int(len(lats) * 0.99)] if lats else 0

    return {
        "total_requests":    _stats["total"],
        "total_violations":  _stats["violations"],
        "total_errors":      _stats["errors"],
        "violation_rate":    round(_stats["violations"] / max(_stats["total"], 1), 4),
        "latency_p50_ms":    round(p50, 2),
        "latency_p95_ms":    round(p95, 2),
        "latency_p99_ms":    round(p99, 2),
    }
