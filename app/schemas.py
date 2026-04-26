from typing import Literal

from pydantic import BaseModel, Field


PredictMode = Literal["classic", "transformer", "hybrid"]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Tweet text to classify")
    mode: PredictMode = Field(default="hybrid")
    hybrid_threshold: float = Field(default=0.78, ge=0.5, le=0.99)
    max_length: int = Field(default=128, ge=16, le=512)
    explain: bool = Field(default=False)
    explanation_top_k: int = Field(default=8, ge=3, le=20)


class PredictResponse(BaseModel):
    label: str
    scores: dict[str, float]
    requested_mode: PredictMode
    model_used: str
    confidence: float
    model_latency_ms: float
    latency_ms: float
    explanation: list[dict[str, float | str]] = Field(default_factory=list)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)
    mode: PredictMode = Field(default="hybrid")
    hybrid_threshold: float = Field(default=0.78, ge=0.5, le=0.99)
    max_length: int = Field(default=128, ge=16, le=512)
    batch_size: int = Field(default=32, ge=1, le=256)


class BatchPredictionItem(BaseModel):
    text: str
    label: str
    confidence: float
    model_used: str
    scores: dict[str, float]
    model_latency_ms: float


class BatchPredictResponse(BaseModel):
    requested_mode: PredictMode
    count: int
    latency_ms: float
    predictions: list[BatchPredictionItem]
