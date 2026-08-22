# Customer Review Sentiment AI Application

End-to-end AI application that classifies e-commerce and food-delivery customer reviews as **positive** or **negative**, with a web UI, JSON API, Docker image, and Azure deployment instructions.

---

## 1. Problem Statement

E-commerce and food-delivery businesses struggle to analyze large volumes of customer reviews quickly. Delayed insight leads to poor product decisions and slow response to complaints. Support and product teams cannot read every comment by hand, so negative feedback is easy to miss.

## 2. Use Case

| Who | How they use it |
|-----|-----------------|
| Product managers | Track product or restaurant quality from incoming reviews |
| Support teams | Prioritize negative reviews for refunds and follow-up |
| Marketing teams | Measure campaign or menu-change sentiment |
| Marketplace sellers | Triage Amazon-style product feedback at scale |

A user pastes a review into the web form (or sends JSON to `/predict`) and receives a sentiment label plus a confidence score in under a second.

## 3. Solution Overview

The solution is a binary text classifier wrapped in a FastAPI application:

1. Review text is cleaned (lowercase, strip HTML/URLs).
2. A TF-IDF vectorizer turns the text into n-gram features.
3. Logistic Regression predicts **negative** or **positive** with class probabilities.
4. Results are returned through a web UI at `/` and a REST API at `/predict`.

On a held-out test set of 4,000 Amazon reviews the model reached **89.65% accuracy** (see [section 5](#5-aiml-approach)).

## 4. Dataset

| Item | Detail |
|------|--------|
| Name | Amazon Polarity (`fancyzhx/amazon_polarity`) |
| Source | [Hugging Face dataset card](https://huggingface.co/datasets/fancyzhx/amazon_polarity) |
| Domain | Amazon product reviews (e-commerce text, including food products) |
| Original size | ~4 million labelled reviews |
| Subset used | 20,000 reviews (balanced: 10,000 negative, 10,000 positive) |
| Split | 15,999 train / 4,000 test, stratified |
| Labels | `0` = negative, `1` = positive |
| Fields used | `title` + `content` concatenated as input text |

`data/sample_reviews.csv` is a small hand-written food-delivery sample used only for UI/API demos. It is **not** the training set. See [`data/README.md`](data/README.md).

Retrain with:

```bash
python -m pip install -r requirements-train.txt
python train/train_model.py
```

## 5. AI/ML Approach

| Item | Choice |
|------|--------|
| Task | Binary text classification |
| Preprocessing | Lowercase, strip HTML and URLs, keep letters/digits |
| Features | TF-IDF, 1–2 grams, 20,000 max features, `sublinear_tf` |
| Model | Logistic Regression (`liblinear`, `C=2.0`) |
| Libraries | `scikit-learn`, `pandas`, `numpy`, `joblib`, `datasets` |
| Artifacts | `models/model.joblib`, `models/vectorizer.joblib` |

### Test metrics (4,000 reviews)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negative | 0.8985 | 0.8940 | 0.8962 | 2,000 |
| Positive | 0.8945 | 0.8990 | 0.8968 | 2,000 |
| **Accuracy** | | | **0.8965** | 4,000 |

Confusion matrix: TN 1788, FP 212, FN 202, TP 1798.

TF-IDF + Logistic Regression is intentionally lightweight so the app starts quickly on Azure App Service and in Docker, without a GPU.

## 6. Application Architecture

```text
Browser / Postman / Swagger
            |
            v
   +----------------------+
   |  FastAPI (one process)
   |  GET  /          web UI
   |  GET  /health
   |  POST /predict
   |  POST /predict/batch
   |  GET  /docs
   +----------------------+
            |
            v
   TF-IDF vectorizer + Logistic Regression
   (models/*.joblib loaded at startup)
```

Docker (and later Azure App Service) runs the same image: Uvicorn serving `app.main:app` on port 8000.

## 7. Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 / 3.12 |
| API | FastAPI, Uvicorn, Pydantic |
| ML | scikit-learn, joblib, pandas, numpy |
| Training data | Hugging Face `datasets` (`fancyzhx/amazon_polarity`) |
| UI | Static HTML/CSS/JS served by FastAPI |
| Container | Docker (Python 3.11-slim) |
| Cloud (when available) | Microsoft Azure App Service (Linux container) or Azure Container Apps |
| Registry | Docker Hub (or Azure Container Registry) |

Runtime dependencies: [`requirements.txt`](requirements.txt). Training extras: [`requirements-train.txt`](requirements-train.txt).

## 8. Local Setup Instructions

Requires **Python 3.11 or 3.12** (macOS default `python3` may be 3.9).

```bash
cd "customer-review-sentiment"

python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Model files are already in models/. To retrain:
# pip install -r requirements-train.txt
# python train/train_model.py

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- Web UI: <http://127.0.0.1:8000/>
- Swagger: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The pasta was delicious and arrived hot. Will order again!"}'
```

More examples: [`sample_requests.http`](sample_requests.http).

## 9. Deployment Details

**Current status:** no Azure subscription was available during development. The assignment fallback is used: the app is Dockerized and can be pushed to Docker Hub, then later deployed to Azure.

### Target Azure services

| Service | Role |
|---------|------|
| Azure App Service (Linux, container) | Host the API + UI |
| Azure Container Registry (optional) | Private image instead of Docker Hub |
| Application Insights (optional) | Request logs and health |

### Deploy from Docker Hub to Azure App Service

Replace placeholders (`RESOURCE_GROUP`, `APP_NAME`, `yourusername`).

```bash
az login

az group create --name rg-review-sentiment --location eastus

az appservice plan create \
  --name plan-review-sentiment \
  --resource-group rg-review-sentiment \
  --is-linux \
  --sku B1

az webapp create \
  --resource-group rg-review-sentiment \
  --plan plan-review-sentiment \
  --name APP_NAME \
  --deployment-container-image-name yourusername/review-sentiment:latest

az webapp config appsettings set \
  --resource-group rg-review-sentiment \
  --name APP_NAME \
  --settings WEBSITES_PORT=8000

# Live URL after deploy:
# https://APP_NAME.azurewebsites.net/
# https://APP_NAME.azurewebsites.net/docs
```

If you use Azure Container Registry instead of Docker Hub:

```bash
az acr create --resource-group rg-review-sentiment --name acrReviewSentiment --sku Basic
az acr login --name acrReviewSentiment
docker tag review-sentiment:latest acrReviewSentiment.azurecr.io/review-sentiment:latest
docker push acrReviewSentiment.azurecr.io/review-sentiment:latest
```

Update this section with the live `*.azurewebsites.net` URL after the first successful deploy.

## 10. API / Web Application Usage

### Web UI

1. Open `/`.
2. Paste a review (or click **Use sample review**).
3. Click **Analyze sentiment**.
4. The page shows the label (positive/negative), confidence, and class probabilities.

### REST API

**`POST /predict`**

Request:

```json
{
  "text": "The pasta was delicious and arrived hot. Will order again!"
}
```

Response:

```json
{
  "sentiment": "positive",
  "confidence": 0.94,
  "probabilities": {
    "negative": 0.06,
    "positive": 0.94
  }
}
```

**`POST /predict/batch`** — up to 50 texts in `{ "texts": ["...", "..."] }`.

**`GET /health`** — `{ "status": "ok", "model_loaded": true }`.

Interactive docs: `/docs` (Swagger UI).

## 11. Docker Instructions

Build and run locally:

```bash
docker build -t review-sentiment:latest .
docker run --rm -p 8000:8000 review-sentiment:latest
```

Then open <http://127.0.0.1:8000/>.

Push to Docker Hub (assignment fallback when Azure is not available):

```bash
docker login
docker tag review-sentiment:latest yourusername/review-sentiment:latest
docker push yourusername/review-sentiment:latest
```

Replace `yourusername` with your Docker Hub account. After the push, record the image URL in this README (for example `docker.io/yourusername/review-sentiment:latest`).

---

## Repository contents

```
customer-review-sentiment/
├── README.md
├── requirements.txt
├── requirements-train.txt
├── Dockerfile
├── .dockerignore
├── sample_requests.http
├── app/
│   ├── main.py
│   ├── predict.py
│   └── schemas.py
├── static/
│   ├── index.html
│   └── styles.css
├── train/
│   └── train_model.py
├── models/
│   ├── model.joblib
│   ├── vectorizer.joblib
│   └── metrics.json
└── data/
    ├── sample_reviews.csv
    └── README.md
```
