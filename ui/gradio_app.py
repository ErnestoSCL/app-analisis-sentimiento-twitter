from __future__ import annotations

import io
import os
from datetime import datetime

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.metrics import classification_report, confusion_matrix


API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_PREDICT = f"{API_BASE}/predict"
API_COMPARE = f"{API_BASE}/predict_compare"
API_HEALTH = f"{API_BASE}/health"
API_HISTORY = f"{API_BASE}/history"


history_rows: list[dict] = []


def _to_scores_df(scores: dict[str, float]) -> pd.DataFrame:
    if not scores:
        return pd.DataFrame(columns=["label", "score"])
    df = pd.DataFrame([{"label": k, "score": v} for k, v in scores.items()])
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def _history_df() -> pd.DataFrame:
    if not history_rows:
        return pd.DataFrame(columns=["timestamp", "text", "mode", "model_used", "label", "confidence", "latency_ms"])
    return pd.DataFrame(history_rows)


def _safe_request(url: str, payload: dict) -> dict:
    r = requests.post(url, json=payload, timeout=240)
    r.raise_for_status()
    return r.json()


def health_panel() -> str:
    try:
        r = requests.get(API_HEALTH, timeout=30)
        r.raise_for_status()
        d = r.json()
        db = d.get("database", {})
        return (
            f"### API Status\n"
            f"- Status: **{d.get('status', 'unknown')}**\n"
            f"- Device: **{d.get('device', 'unknown')}**\n"
            f"- GPU: **{d.get('gpu_name', 'n/a')}**\n"
            f"- Classic model: `{d.get('classic_model', 'n/a')}` (F1: {d.get('classic_test_f1', 'n/a')})\n"
            f"- Transformer model: `{d.get('transformer_model', 'n/a')}` (F1: {d.get('transformer_test_f1', 'n/a')})\n"
            f"- Database: **{db.get('status', 'unknown')}**"
        )
    except Exception as e:
        return f"### API Status\n- Status: **offline/error**\n- Detail: `{e}`"


def single_predict(
    text: str,
    mode: str,
    threshold: float,
    max_length: int,
    explain: bool,
    top_k: int,
):
    txt = (text or "").strip()
    if not txt:
        raise gr.Error("Please provide a tweet text.")

    payload = {
        "text": txt,
        "mode": mode,
        "hybrid_threshold": float(threshold),
        "max_length": int(max_length),
        "explain": bool(explain),
        "explanation_top_k": int(top_k),
    }

    data = _safe_request(API_PREDICT, payload)
    scores_df = _to_scores_df(data.get("scores", {}))
    exp_df = pd.DataFrame(data.get("explanation", []))

    summary = (
        f"### Prediction\n"
        f"- Label: **{data['label']}**\n"
        f"- Requested mode: `{data['requested_mode']}`\n"
        f"- Model used: `{data['model_used']}`\n"
        f"- Confidence: **{data['confidence']:.4f}**\n"
        f"- Model latency: **{data.get('model_latency_ms', 0):.2f} ms**\n"
        f"- Total latency: **{data['latency_ms']:.2f} ms**"
    )

    history_rows.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "text": txt,
            "mode": mode,
            "model_used": data.get("model_used"),
            "label": data.get("label"),
            "confidence": round(float(data.get("confidence", 0.0)), 4),
            "latency_ms": round(float(data.get("latency_ms", 0.0)), 2),
        }
    )

    hdf = _history_df()
    return summary, scores_df, exp_df, hdf


def compare_predict(text: str, threshold: float, max_length: int):
    txt = (text or "").strip()
    if not txt:
        raise gr.Error("Please provide a tweet text.")

    payload = {
        "text": txt,
        "mode": "hybrid",
        "hybrid_threshold": float(threshold),
        "max_length": int(max_length),
    }
    data = _safe_request(API_COMPARE, payload)
    preds = data.get("predictions", {})

    rows = []
    for mode in ["classic", "transformer", "hybrid"]:
        p = preds.get(mode, {})
        rows.append(
            {
                "mode": mode,
                "model_used": p.get("model_used", "n/a"),
                "label": p.get("label", "n/a"),
                "confidence": round(float(p.get("confidence", 0.0)), 4),
                "model_latency_ms": float(p.get("model_latency_ms", 0.0)),
            }
        )
    cmp_df = pd.DataFrame(rows)
    summary = f"### Compare complete\n- Total compare latency: **{data.get('latency_ms', 0)} ms**"
    return summary, cmp_df


def export_history_csv() -> tuple[str | None, str]:
    hdf = _history_df()
    if hdf.empty:
        return None, "No history to export yet."
    path = "ui_prediction_history.csv"
    hdf.to_csv(path, index=False)
    return path, f"Exported {len(hdf)} rows to `{path}`"


def clear_history() -> pd.DataFrame:
    history_rows.clear()
    return _history_df()


def refresh_db_history(limit: int):
    try:
        r = requests.get(API_HISTORY, params={"limit": int(limit)}, timeout=60)
        r.raise_for_status()
        items = r.json().get("items", [])
        return pd.DataFrame(items)
    except Exception as e:
        raise gr.Error(f"Could not fetch DB history: {e}")


def batch_error_analysis(file_obj):
    if file_obj is None:
        raise gr.Error("Upload a CSV with columns: text, true_label")

    df = pd.read_csv(file_obj.name)
    required = {"text", "true_label"}
    if not required.issubset(df.columns):
        raise gr.Error("CSV must include columns: text, true_label")

    preds = []
    for text in df["text"].astype(str).tolist():
        payload = {
            "text": text,
            "mode": "hybrid",
            "hybrid_threshold": 0.78,
            "max_length": 128,
            "explain": False,
            "explanation_top_k": 8,
        }
        out = _safe_request(API_PREDICT, payload)
        preds.append(out["label"])

    eval_df = df.copy()
    eval_df["pred_label"] = preds

    report = classification_report(eval_df["true_label"], eval_df["pred_label"], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})

    labels = sorted(set(eval_df["true_label"]).union(set(eval_df["pred_label"])))
    cm = confusion_matrix(eval_df["true_label"], eval_df["pred_label"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_df.values)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, str(cm_df.values[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    return report_df, cm_df, fig, eval_df.head(20)


THEME_CSS = """
.gradio-container {max-width: 1200px !important; margin: 0 auto !important;}
"""


with gr.Blocks(theme=gr.themes.Soft(), css=THEME_CSS, title="Twitter Sentiment Hybrid Pro") as demo:
    gr.Markdown("# Twitter Sentiment Hybrid Pro")
    gr.Markdown("**Tagline:** FastAPI + Hybrid Routing + Explainability + Batch Analytics")

    with gr.Row():
        health_md = gr.Markdown(health_panel())

    with gr.Tab("Live Inference"):
        with gr.Row():
            text = gr.Textbox(lines=4, label="Tweet text", placeholder="Type a tweet...")

        with gr.Accordion("Advanced Mode", open=False):
            with gr.Row():
                mode = gr.Dropdown(["hybrid", "classic", "transformer"], value="hybrid", label="Mode")
                threshold = gr.Slider(0.5, 0.99, value=0.78, step=0.01, label="Hybrid threshold")
                max_length = gr.Slider(16, 512, value=128, step=8, label="Max length")
                top_k = gr.Slider(3, 20, value=8, step=1, label="Explanation top-k")
                explain = gr.Checkbox(value=False, label="Enable explanation")

        with gr.Row():
            run_btn = gr.Button("Predict", variant="primary")
            refresh_health = gr.Button("Refresh Health")

        pred_summary = gr.Markdown()
        gr.Markdown("### Confidence by Class")
        score_table = gr.Dataframe(headers=["label", "score"], datatype=["str", "number"], interactive=False)
        gr.Markdown("### Explainability (top features)")
        exp_table = gr.Dataframe(headers=["feature", "contribution"], datatype=["str", "number"], interactive=False)

    with gr.Tab("A/B/C Compare"):
        cmp_text = gr.Textbox(lines=4, label="Tweet text", placeholder="Type a tweet for mode comparison...")
        with gr.Row():
            cmp_threshold = gr.Slider(0.5, 0.99, value=0.78, step=0.01, label="Hybrid threshold")
            cmp_max_len = gr.Slider(16, 512, value=128, step=8, label="Max length")
            cmp_btn = gr.Button("Run Compare", variant="primary")

        cmp_summary = gr.Markdown()
        cmp_table = gr.Dataframe(
            headers=["mode", "model_used", "label", "confidence", "model_latency_ms"],
            datatype=["str", "str", "str", "number", "number"],
            interactive=False,
        )

    with gr.Tab("History"):
        history_table = gr.Dataframe(
            headers=["timestamp", "text", "mode", "model_used", "label", "confidence", "latency_ms"],
            datatype=["str", "str", "str", "str", "str", "number", "number"],
            interactive=False,
        )
        with gr.Row():
            export_btn = gr.Button("Export CSV")
            clear_btn = gr.Button("Clear History")
            db_limit = gr.Slider(10, 500, value=100, step=10, label="DB history limit")
            db_history_btn = gr.Button("Load From DB")
        db_history_table = gr.Dataframe(label="History persisted in MySQL")
        export_file = gr.File(label="Exported file")
        export_msg = gr.Markdown()

    with gr.Tab("Batch Error Analysis"):
        gr.Markdown("Upload CSV with columns: `text`, `true_label`")
        batch_file = gr.File(file_types=[".csv"], label="Input CSV")
        batch_btn = gr.Button("Analyze", variant="primary")
        report_df_out = gr.Dataframe(label="Classification Report")
        cm_df_out = gr.Dataframe(label="Confusion Matrix")
        cm_plot = gr.Plot(label="Confusion Matrix Plot")
        sample_eval = gr.Dataframe(label="Sample predictions")

    run_btn.click(
        single_predict,
        inputs=[text, mode, threshold, max_length, explain, top_k],
        outputs=[pred_summary, score_table, exp_table, history_table],
    )
    refresh_health.click(lambda: health_panel(), outputs=[health_md])
    cmp_btn.click(compare_predict, inputs=[cmp_text, cmp_threshold, cmp_max_len], outputs=[cmp_summary, cmp_table])
    export_btn.click(export_history_csv, outputs=[export_file, export_msg])
    clear_btn.click(clear_history, outputs=[history_table])
    db_history_btn.click(refresh_db_history, inputs=[db_limit], outputs=[db_history_table])
    batch_btn.click(batch_error_analysis, inputs=[batch_file], outputs=[report_df_out, cm_df_out, cm_plot, sample_eval])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
