# Twitter Sentiment Analysis App (Hybrid AI)

Production-ready Twitter sentiment application with a **hybrid inference strategy**:

- **Classic path**: TF-IDF + Logistic Regression (low latency, low cost)
- **Transformer path**: fine-tuned RoBERTa (higher predictive quality)
- **Hybrid router**: runs classic first and falls back to transformer when confidence is below threshold

---

## Dataset

- Kaggle source: https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset/
- Task: 3-class sentiment classification (`negative`, `neutral`, `positive`)

---

## Technology Stack

### Machine Learning / NLP
- **scikit-learn**: TF-IDF feature engineering and linear classification
- **Hugging Face Transformers**: fine-tuned RoBERTa sequence classification
- **PyTorch**: transformer inference runtime
- **NumPy / pandas**: numerical and tabular processing

### Backend
- **FastAPI**: REST API and interactive OpenAPI docs
- **Pydantic**: request/response validation
- **SQLAlchemy 2.x**: ORM and persistence layer
- **PyMySQL**: MySQL DB driver

### Frontend / UX
- **Gradio**: professional interactive UI with:
  - live inference
  - A/B/C compare
  - explainability views
  - batch error analysis
  - history export

### Infrastructure
- **Docker** + **Docker Compose**: container orchestration for API, UI, and MySQL

---

## Core Features

- Single-text prediction (`classic`, `transformer`, `hybrid`)
- Advanced inference controls (hybrid threshold, max sequence length, explain mode)
- Explainability:
  - Classic: top TF-IDF feature contributions
  - Transformer: attention-based token importance approximation
- Compare endpoint/UI for `classic` vs `transformer` vs `hybrid`
- Batch prediction endpoint
- Batch error analysis (classification report + confusion matrix)
- Persistent prediction logging in MySQL
- Health/status endpoint with model, runtime device, and DB state

---

## High-Level Architecture

```text
Gradio UI ---> FastAPI API ---> Hybrid Router ---> Classic Model (TF-IDF + LR)
                                 |                
                                 +--------------> Transformer (fine-tuned RoBERTa)
                                 |
                                 +--------------> MySQL (prediction logs)
```

---

## Professional Project Structure

```text
app/
  __init__.py
  main.py                 # API routes
  schemas.py              # Pydantic contracts
  service.py              # inference + explainability + routing
  db.py                   # SQLAlchemy models and DB helpers

ui/
  gradio_app.py           # interactive web interface

artifacts/
  models/
    classic/
    transformer/
  reports/
  predictions/

notebooks/
  Analisis_Sentimientos_Twitter_EDA_Procesamiento.ipynb

infra/
  Dockerfile
  docker-compose.yml

requirements.txt
README.md
```

---

## Local Development

### 1) Start API

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
set DATABASE_URL=mysql+pymysql://app_user:app_password@localhost:3306/sentiment_app
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2) Start UI (new terminal)

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
docker compose -f infra/docker-compose.yml up --build
```

Services:
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- UI: `http://localhost:7860`
- MySQL: `localhost:3306`

---

## Main API Endpoints

- `GET /health`
- `GET /history`
- `POST /predict`
- `POST /predict_compare`
- `POST /batch_predict`

### `POST /predict` Example

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

CSV columns required:
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
- Current Docker image runs CPU inference by default.
- For GPU containers, use a CUDA base image + NVIDIA Container Toolkit.
