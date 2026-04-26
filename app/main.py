from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.db import fetch_recent_predictions, get_db_status, init_db, save_prediction_log

from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BatchPredictionItem,
    PredictRequest,
    PredictResponse,
)
from app.service import SentimentService


app = FastAPI(
    title="Twitter Sentiment API",
    version="1.0.0",
    description="Hybrid sentiment API: classical TF-IDF and fine-tuned RoBERTa",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service: SentimentService | None = None


@app.on_event("startup")
def on_startup() -> None:
    global service
    root = Path(__file__).resolve().parents[1]
    init_db()
    service = SentimentService(root)


@app.get("/health")
def health() -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    gpu_name = "cpu"
    if service.device == "cuda":
        import torch

        gpu_name = torch.cuda.get_device_name(0)

    return {
        "status": "ok",
        "device": service.device,
        "gpu_name": gpu_name,
        "classic_model": service.classic_metadata.get("model_name", "loaded"),
        "transformer_model": service.transformer_metadata.get("model_name", "loaded"),
        "classic_test_f1": service.classic_metadata.get("test_f1_macro"),
        "transformer_test_f1": service.transformer_metadata.get("test_f1_macro"),
        "database": get_db_status(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    t0 = time.perf_counter()
    if req.mode == "classic":
        out = service.predict_classic(text, explain=req.explain, top_k=req.explanation_top_k)
    elif req.mode == "transformer":
        out = service.predict_transformer(
            text,
            max_length=req.max_length,
            explain=req.explain,
            top_k=req.explanation_top_k,
        )
    else:
        out = service.predict_hybrid(
            text,
            req.hybrid_threshold,
            max_length=req.max_length,
            explain=req.explain,
            top_k=req.explanation_top_k,
        )

    total_latency_ms = (time.perf_counter() - t0) * 1000

    save_prediction_log(
        text_value=text,
        requested_mode=req.mode,
        model_used=out["model_used"],
        label=out["label"],
        confidence=float(out["confidence"]),
        model_latency_ms=float(out["model_latency_ms"]),
        total_latency_ms=round(total_latency_ms, 2),
    )

    return PredictResponse(
        label=out["label"],
        scores=out["scores"],
        requested_mode=req.mode,
        model_used=out["model_used"],
        confidence=float(out["confidence"]),
        model_latency_ms=float(out["model_latency_ms"]),
        latency_ms=round(total_latency_ms, 2),
        explanation=out.get("explanation", []),
    )


@app.post("/predict_compare")
def predict_compare(req: PredictRequest) -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    t0 = time.perf_counter()
    outputs = service.predict_all_modes(text, threshold=req.hybrid_threshold, max_length=req.max_length)
    total_latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "text": text,
        "latency_ms": round(total_latency_ms, 2),
        "predictions": outputs,
    }


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(req: BatchPredictRequest) -> BatchPredictResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    clean_texts = [t.strip() for t in req.texts if t and t.strip()]
    if not clean_texts:
        raise HTTPException(status_code=400, detail="No valid texts provided")

    t0 = time.perf_counter()
    out_items: list[BatchPredictionItem] = []
    for text in clean_texts:
        if req.mode == "classic":
            pred = service.predict_classic(text)
        elif req.mode == "transformer":
            pred = service.predict_transformer(text, max_length=req.max_length)
        else:
            pred = service.predict_hybrid(text, threshold=req.hybrid_threshold, max_length=req.max_length)

        out_items.append(
            BatchPredictionItem(
                text=text,
                label=str(pred["label"]),
                confidence=float(pred["confidence"]),
                model_used=str(pred["model_used"]),
                scores=pred["scores"],
                model_latency_ms=float(pred["model_latency_ms"]),
            )
        )

    total_latency_ms = (time.perf_counter() - t0) * 1000
    return BatchPredictResponse(
        requested_mode=req.mode,
        count=len(out_items),
        latency_ms=round(total_latency_ms, 2),
        predictions=out_items,
    )


@app.get("/history")
def history(limit: int = 100) -> dict:
    limit = max(1, min(limit, 1000))
    return {"items": fetch_recent_predictions(limit=limit)}
