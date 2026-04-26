from __future__ import annotations

import os
import time
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://app_user:app_password@db:3306/sentiment_app",
)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    requested_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    model_used: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str] = mapped_column(String(32), nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)


engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    last_err = None
    for _ in range(30):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except Exception as e:
            last_err = e
            time.sleep(2)
    if last_err:
        raise last_err


def get_db_status() -> dict:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "url": DATABASE_URL}
    except Exception as e:
        return {"status": "error", "error": str(e), "url": DATABASE_URL}


def save_prediction_log(
    text_value: str,
    requested_mode: str,
    model_used: str,
    label: str,
    confidence: float,
    model_latency_ms: float,
    total_latency_ms: float,
) -> None:
    session: Session = SessionLocal()
    try:
        row = PredictionLog(
            text=text_value,
            requested_mode=requested_mode,
            model_used=model_used,
            label=label,
            confidence=confidence,
            model_latency_ms=model_latency_ms,
            total_latency_ms=total_latency_ms,
        )
        session.add(row)
        session.commit()
    finally:
        session.close()


def fetch_recent_predictions(limit: int = 100) -> list[dict]:
    session: Session = SessionLocal()
    try:
        rows = (
            session.query(PredictionLog)
            .order_by(PredictionLog.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "text": r.text,
                "requested_mode": r.requested_mode,
                "model_used": r.model_used,
                "label": r.label,
                "confidence": r.confidence,
                "model_latency_ms": r.model_latency_ms,
                "total_latency_ms": r.total_latency_ms,
            }
            for r in rows
        ]
    finally:
        session.close()
