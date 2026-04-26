# Twitter Sentiment Analysis App (Hybrid AI)

Production-style Twitter sentiment application with a hybrid inference strategy:

- **Classic**: TF-IDF + Logistic Regression (fast and low-cost)
- **Transformer**: fine-tuned RoBERTa (higher accuracy)
- **Hybrid Router**: uses classic first; if confidence is low, falls back to transformer

This project includes:

- FastAPI backend
- Gradio professional UI (live inference, compare, explainability, batch analysis)
- MySQL persistence for prediction logs
- Docker and Docker Compose orchestration

Dataset used:

- https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset/

---

## Key Features

- Live single-text prediction (`classic`, `transformer`, `hybrid`)
- Advanced controls (threshold, max token length, explanations)
- Explainability:
  - Classic model feature contributions
  - Transformer attention-based token importance approximation
- A/B/C compare endpoint and UI panel
- Batch prediction endpoint
- Batch error analysis tool (confusion matrix + classification report)
- Persistent prediction history in MySQL
- Health endpoint with model/device/database status

---

## Architecture

```text
Gradio UI ---> FastAPI API ---> Hybrid Router ---> Classic Model (TF-IDF + LR)
                                 |                
                                 +--------------> Transformer (fine-tuned RoBERTa)
                                 |
                                 +--------------> MySQL (prediction logs)
```

---

## Project Structure

```text
app/
  main.py            # API routes
  schemas.py         # Pydantic request/response models
  service.py         # Inference + explainability + hybrid routing
  db.py              # SQLAlchemy + MySQL persistence
ui/
  gradio_app.py      # Professional UI
artifacts/
  models/
    classic/
    transformer/
Dockerfile
docker-compose.yml
requirements.txt
```

---

## Local Development

### 1) Backend

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
set DATABASE_URL=mysql+pymysql://app_user:app_password@localhost:3306/sentiment_app
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2) UI (new terminal)

```powershell
.\.venv\Scripts\Activate.ps1
set API_BASE_URL=http://localhost:8000
python ui/gradio_app.py
```

- API docs: `http://localhost:8000/docs`
- UI: `http://localhost:7860`

---

## Dockerized Run (API + UI + MySQL)

```powershell
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- UI: `http://localhost:7860`
- MySQL: `localhost:3306`

---

## Main API Endpoints

- `GET /health`
- `POST /predict`
- `POST /predict_compare`
- `POST /batch_predict`
- `GET /history`

### `POST /predict` example

```json
{
  "text": "I love this phone, battery is amazing!",
  "mode": "hybrid",
  "hybrid_threshold": 0.78,
  "max_length": 128,
  "explain": true,
  "explanation_top_k": 8
}
```

---

## Batch Error Analysis Input

Upload CSV with columns:

- `text`
- `true_label`

Example:

```csv
text,true_label
I love this!,positive
This is terrible,negative
I went to work,neutral
```

---

## Notes

- Trained model artifacts are expected under `artifacts/models/...`.
- Docker image currently runs CPU inference by default.
- For GPU in Docker, use a CUDA base image and NVIDIA Container Toolkit.
