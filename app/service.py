from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def preprocess_classic(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " @user ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "__empty__"


def preprocess_transformer(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


class SentimentService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.artifacts_dir = project_root / "artifacts"
        self.classic_model_path = self.artifacts_dir / "models" / "classic" / "best_classic_pipeline.pkl"
        self.transformer_dir = self.artifacts_dir / "models" / "transformer" / "twitter_roberta_finetuned"
        self.classic_meta_path = self.artifacts_dir / "models" / "classic" / "metadata.json"
        self.transformer_meta_path = self.artifacts_dir / "models" / "transformer" / "metadata.json"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.classic_model = joblib.load(self.classic_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_dir)
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(self.transformer_dir).to(self.device)
        self.transformer_model.eval()

        cfg = self.transformer_model.config
        self.id2label = {int(k): v for k, v in cfg.id2label.items()}
        self.labels = [self.id2label[i] for i in sorted(self.id2label)]

        self.classic_metadata = self._load_json(self.classic_meta_path)
        self.transformer_metadata = self._load_json(self.transformer_meta_path)

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalize_scores(scores: dict[str, float], labels: list[str]) -> dict[str, float]:
        out = {lbl: float(scores.get(lbl, 0.0)) for lbl in labels}
        total = sum(out.values())
        if total > 0:
            out = {k: v / total for k, v in out.items()}
        return out

    def explain_classic(self, text: str, top_k: int = 8) -> list[dict[str, float | str]]:
        if not hasattr(self.classic_model, "named_steps"):
            return []
        clf = self.classic_model.named_steps.get("clf")
        features = self.classic_model.named_steps.get("features")
        if clf is None or features is None or not hasattr(clf, "coef_"):
            return []
        try:
            pre = preprocess_classic(text)
            vec = features.transform([pre])
            pred = self.classic_model.predict([pre])[0]
            classes = list(clf.classes_)
            cls_idx = classes.index(pred)
            coefs = clf.coef_[cls_idx]
            feat_names = features.get_feature_names_out()
            row = vec.tocoo()

            contrib = []
            for idx, value in zip(row.col, row.data):
                weight = float(coefs[idx])
                contrib.append((feat_names[idx], float(value * weight)))
            contrib.sort(key=lambda x: abs(x[1]), reverse=True)
            return [{"feature": str(name), "contribution": float(score)} for name, score in contrib[:top_k]]
        except Exception:
            return []

    def explain_transformer(self, text: str, top_k: int = 8, max_length: int = 128) -> list[dict[str, float | str]]:
        clean = preprocess_transformer(text)
        try:
            enc = self.tokenizer(
                clean,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.transformer_model(**enc, output_attentions=True)
            attn = out.attentions[-1][0]
            token_importance = attn.mean(dim=0).mean(dim=0).detach().cpu().numpy()
            token_ids = enc["input_ids"][0].detach().cpu().numpy().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            pairs = []
            for tok, score in zip(tokens, token_importance):
                if tok in {"<s>", "</s>", "<pad>"}:
                    continue
                clean_tok = tok.replace("Ġ", "").strip()
                if not clean_tok:
                    continue
                pairs.append((clean_tok, float(score)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return [{"feature": str(tok), "contribution": float(score)} for tok, score in pairs[:top_k]]
        except Exception:
            return []

    def predict_classic(self, text: str, explain: bool = False, top_k: int = 8) -> dict:
        x = [preprocess_classic(text)]
        t0 = time.perf_counter()

        if hasattr(self.classic_model, "predict_proba"):
            probs = self.classic_model.predict_proba(x)[0]
            classes = list(self.classic_model.classes_)
            scores = {str(c): float(p) for c, p in zip(classes, probs)}
            label = classes[int(np.argmax(probs))]
            confidence = float(np.max(probs))
        else:
            label = self.classic_model.predict(x)[0]
            scores = {str(lbl): 0.0 for lbl in self.labels}
            scores[str(label)] = 1.0
            confidence = 1.0

        model_latency_ms = (time.perf_counter() - t0) * 1000
        explanation = self.explain_classic(text, top_k=top_k) if explain else []
        return {
            "label": str(label),
            "scores": self._normalize_scores(scores, self.labels),
            "confidence": confidence,
            "model_latency_ms": round(model_latency_ms, 2),
            "model_used": "classic",
            "explanation": explanation,
        }

    def predict_transformer(self, text: str, max_length: int = 128, explain: bool = False, top_k: int = 8) -> dict:
        clean = preprocess_transformer(text)
        t0 = time.perf_counter()

        enc = self.tokenizer(clean, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.transformer_model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        scores = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}
        best_idx = int(np.argmax(probs))
        label = self.id2label[best_idx]
        confidence = float(np.max(probs))
        model_latency_ms = (time.perf_counter() - t0) * 1000
        explanation = self.explain_transformer(text, top_k=top_k, max_length=max_length) if explain else []

        return {
            "label": str(label),
            "scores": self._normalize_scores(scores, self.labels),
            "confidence": confidence,
            "model_latency_ms": round(model_latency_ms, 2),
            "model_used": "transformer",
            "explanation": explanation,
        }

    def predict_hybrid(
        self,
        text: str,
        threshold: float,
        max_length: int = 128,
        explain: bool = False,
        top_k: int = 8,
    ) -> dict:
        classic = self.predict_classic(text, explain=explain, top_k=top_k)
        if classic["confidence"] >= threshold:
            classic["model_used"] = "hybrid->classic"
            return classic

        transformer = self.predict_transformer(text, max_length=max_length, explain=explain, top_k=top_k)
        transformer["model_used"] = "hybrid->transformer"
        return transformer

    def predict_all_modes(self, text: str, threshold: float, max_length: int = 128) -> dict[str, dict]:
        classic = self.predict_classic(text)
        transformer = self.predict_transformer(text, max_length=max_length)
        hybrid = self.predict_hybrid(text, threshold=threshold, max_length=max_length)
        return {
            "classic": classic,
            "transformer": transformer,
            "hybrid": hybrid,
        }
