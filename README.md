# Customer Review Sentiment AI Application

**Live application:** [https://review-sentiment-msdjshan.azurewebsites.net](https://review-sentiment-msdjshan.azurewebsites.net)

End-to-end AI application that classifies e-commerce and food-delivery customer reviews as **positive** or **negative**. It includes the **Signal** web UI, a JSON API, a Docker image, and a live Azure App Service deployment.

**Public repository:** https://github.com/DonShan/customer-review-sentiment

| Link | URL |
|------|-----|
| Web UI (Signal) | https://review-sentiment-msdjshan.azurewebsites.net |
| Swagger API docs | https://review-sentiment-msdjshan.azurewebsites.net/docs |
| Health check | https://review-sentiment-msdjshan.azurewebsites.net/health |
| Sample dataset | https://review-sentiment-msdjshan.azurewebsites.net/sample-data |
| Docker image | `madushansenavirathna/review-sentiment:latest` (`linux/amd64`) |

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

Users analyze one review or a batch (up to 50) in the Signal UI, or call `/predict` and `/predict/batch` from another system.

## 3. Solution Overview

The solution is a binary text classifier wrapped in a FastAPI application branded as **Signal**:

1. Review text is cleaned (lowercase, strip HTML/URLs).
2. A TF-IDF vectorizer turns the text into n-gram features.
3. Logistic Regression predicts **negative** or **positive** with class probabilities.
4. The Signal UI shows the label, confidence, recommended action, and probability bars.
5. Batch mode analyzes many reviews, summarizes counts, and can export CSV.

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

`data/sample_reviews.csv` is a small hand-written food-delivery sample used in the UI (Load sample dataset) and `/sample-data`. It is **not** the training set. See [`data/README.md`](data/README.md).

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
Browser (Signal UI) / Postman / Swagger
            |
            v
   +--------------------------------+
   |  FastAPI (one process)         |
   |  GET  /                 Signal UI
   |  GET  /sample-data      demo CSV
   |  GET  /health
   |  POST /predict
   |  POST /predict/batch
   |  GET  /docs
   +--------------------------------+
            |
            v
   TF-IDF vectorizer + Logistic Regression
   (models/*.joblib loaded at startup)
```

Docker and Azure App Service run the same `linux/amd64` image: Uvicorn serving `app.main:app` on port 8000.

## 7. Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11 / 3.12 |
| API | FastAPI, Uvicorn, Pydantic |
| ML | scikit-learn, joblib, pandas, numpy |
| Training data | Hugging Face `datasets` (`fancyzhx/amazon_polarity`) |
| UI | Signal — static HTML/CSS/JS (`index.html`, `styles.css`, `live.css`, `app.js`) |
| Container | Docker (`python:3.11-slim`, `linux/amd64`) |
| Cloud | Microsoft Azure App Service (Linux container, West US 2, B1) |
| Registry | Docker Hub (`madushansenavirathna/review-sentiment`) |

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
- Sample CSV: <http://127.0.0.1:8000/sample-data>

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The pasta was delicious and arrived hot. Will order again!"}'
```

More examples: [`sample_requests.http`](sample_requests.http).

## 9. Deployment Details

**Current status:** live on Azure App Service. Latest Signal UI image was rebuilt and pushed on 22 Aug 2026.

| Item | Value |
|------|--------|
| Live UI | https://review-sentiment-msdjshan.azurewebsites.net/ |
| API docs | https://review-sentiment-msdjshan.azurewebsites.net/docs |
| Health | https://review-sentiment-msdjshan.azurewebsites.net/health |
| Resource group | `rg-review-sentiment` |
| App Service plan | `plan-review-sentiment-westus2` (B1 Linux) |
| Web app | `review-sentiment-msdjshan` |
| Image | `madushansenavirathna/review-sentiment:latest` (`linux/amd64`) |
| Azure account | `msdjshan47@gmail.com` |

The Docker Hub repository is private (plan limits). App Service uses registry credentials and `WEBSITES_PORT=8000` so it can pull the image.

### Target Azure services

| Service | Role |
|---------|------|
| Azure App Service (Linux, container) | Host the Signal UI + API |
| Docker Hub | Store the `linux/amd64` image |
| Azure Container Registry (optional) | Private alternative to Docker Hub |

### Deploy / update the live site

```bash
az login

# First-time create (already done)
az group create --name rg-review-sentiment --location westus2
az appservice plan create \
  --name plan-review-sentiment-westus2 \
  --resource-group rg-review-sentiment \
  --location westus2 \
  --is-linux \
  --sku B1
az webapp create \
  --resource-group rg-review-sentiment \
  --plan plan-review-sentiment-westus2 \
  --name review-sentiment-msdjshan \
  --deployment-container-image-name madushansenavirathna/review-sentiment:latest
az webapp config appsettings set \
  --resource-group rg-review-sentiment \
  --name review-sentiment-msdjshan \
  --settings WEBSITES_PORT=8000

# Update after code changes
docker build --platform linux/amd64 -t madushansenavirathna/review-sentiment:latest .
docker push madushansenavirathna/review-sentiment:latest
az webapp restart --name review-sentiment-msdjshan --resource-group rg-review-sentiment
```

East US B1 was rejected on this free subscription (0 VM quota). West US 2 succeeded.

If SCM log download (`az webapp log download`) returns **504 Gateway Timeout**, the site is usually still starting or failing to pull the image. Check container logs instead:

```bash
az rest --method POST \
  --uri "/subscriptions/<sub-id>/resourceGroups/rg-review-sentiment/providers/Microsoft.Web/sites/review-sentiment-msdjshan/containerlogs?api-version=2023-12-01"
```

## 10. API / Web Application Usage

### Signal web UI

1. Open https://review-sentiment-msdjshan.azurewebsites.net/ (or `/` locally).
2. **Single review:** paste text or click a sample (Great product / Poor delivery / Helpful support), then **Analyze sentiment**.
3. The page shows sentiment, confidence, a recommended action, and probability bars.
4. **Batch analysis:** switch tabs, paste one review per line (up to 50), or **Load sample dataset**.
5. Review the summary cards, filter rows, and **Export CSV** if needed.

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
  "confidence": 0.7691,
  "probabilities": {
    "negative": 0.2309,
    "positive": 0.7691
  }
}
```

**`POST /predict/batch`** — up to 50 texts in `{ "texts": ["...", "..."] }`.

**`GET /health`** — `{ "status": "ok", "model_loaded": true }`.

**`GET /sample-data`** — downloads `sample_reviews.csv`.

Interactive docs: `/docs` (Swagger UI).

## 11. Docker Instructions

Build for Azure-compatible **linux/amd64** (required on Apple Silicon):

```bash
docker build --platform linux/amd64 -t review-sentiment:latest .
docker run --rm -p 8000:8000 review-sentiment:latest
```

Then open <http://127.0.0.1:8000/>.

Push to Docker Hub:

```bash
docker login
docker tag review-sentiment:latest madushansenavirathna/review-sentiment:latest
docker push madushansenavirathna/review-sentiment:latest
```

Published image: `docker.io/madushansenavirathna/review-sentiment:latest`.

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
│   ├── styles.css
│   ├── live.css
│   └── app.js
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
