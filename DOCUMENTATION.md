# Project Documentation

# Signal — Customer Review Sentiment AI Application

**Course assignment:** End-to-End AI Application Development and Cloud Deployment  
**Student repository:** https://github.com/DonShan/customer-review-sentiment  
**Live application:** https://review-sentiment-msdjshan.azurewebsites.net  
**Azure account:** msdjshan47@gmail.com  
**Docker image:** `madushansenavirathna/review-sentiment:latest` (`linux/amd64`)

This document explains the problem, the AI solution, how the application works, how it was deployed on Microsoft Azure, and how to run, test, and maintain it.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Problem statement](#2-problem-statement)
3. [Objectives and assignment coverage](#3-objectives-and-assignment-coverage)
4. [Use cases](#4-use-cases)
5. [Solution overview](#5-solution-overview)
6. [Dataset](#6-dataset)
7. [AI and machine-learning approach](#7-ai-and-machine-learning-approach)
8. [System architecture](#8-system-architecture)
9. [Application features](#9-application-features)
10. [API reference](#10-api-reference)
11. [Technology stack](#11-technology-stack)
12. [Project structure](#12-project-structure)
13. [Local setup](#13-local-setup)
14. [Docker](#14-docker)
15. [Azure cloud deployment](#15-azure-cloud-deployment)
16. [Cost and budget](#16-cost-and-budget)
17. [Testing and evaluation](#17-testing-and-evaluation)
18. [Limitations and future work](#18-limitations-and-future-work)
19. [How to start and stop the live site](#19-how-to-start-and-stop-the-live-site)
20. [References](#20-references)

---

## 1. Executive summary

Signal is a complete AI web application that reads a customer review and classifies it as **positive** or **negative**. It is built for e-commerce and food-delivery teams that receive more comments than they can read by hand.

The system includes:

- A trained machine-learning model (TF-IDF + Logistic Regression)
- A FastAPI backend with JSON endpoints
- A browser UI (Signal) for single-review and batch analysis
- A Docker image that runs the same app locally or in the cloud
- A public GitHub repository
- A live deployment on **Azure App Service** (Linux container, West US 2)

On a 4,000-review test set the model reached **89.65% accuracy**. Inference typically completes in under one second and does not need a GPU.

---

## 2. Problem statement

E-commerce and food-delivery businesses collect large volumes of reviews from apps, marketplaces, and support tickets. Manual reading is slow. By the time a manager notices a pattern, customers may already have requested refunds or left.

Typical consequences:

- Negative reviews are not routed to support quickly
- Product or restaurant quality issues stay hidden
- Marketing cannot measure whether a campaign or menu change helped
- Sellers on marketplaces cannot triage feedback at scale

The project treats this as a **text classification** problem: given review text, predict sentiment and a confidence score so teams can act immediately.

---

## 3. Objectives and assignment coverage

The assignment required an independently chosen AI use case, a working application (UI or API), cloud deployment, source control, and a documented README. This project covers every item.

| Assignment requirement | How this project meets it |
|------------------------|---------------------------|
| Identify an AI use case | Customer review sentiment for e-commerce / food delivery |
| Public dataset | Hugging Face `fancyzhx/amazon_polarity` (20k subset) |
| End-to-end application | Signal web UI + REST API |
| Cloud deployment | Azure App Service (Linux container) |
| Containerization | Dockerfile; image on Docker Hub |
| Public Git repository | https://github.com/DonShan/customer-review-sentiment |
| README with 11 required sections | See [`README.md`](README.md) |

---

## 4. Use cases

| User | Task |
|------|------|
| Product manager | Watch sentiment after a product or menu change |
| Support lead | Sort incoming comments and escalate strong negatives |
| Marketing | Measure campaign or delivery-experience feedback |
| Marketplace seller | Batch-score a list of Amazon-style reviews |
| Developer | Call `/predict` from another system |

**Example:** A food-delivery operator pastes fifty recent comments into Batch analysis. Signal reports how many are negative, highlights low-confidence cases for manual review, and exports a CSV for the operations team.

---

## 5. Solution overview

Signal is a **binary sentiment classifier** packaged as one web process.

1. The user submits one review or a list of reviews.
2. Text is cleaned (lowercase, HTML and URLs removed).
3. A TF-IDF vectorizer turns text into numeric features (unigrams and bigrams).
4. Logistic Regression outputs class probabilities for **negative** and **positive**.
5. The API returns `{ sentiment, confidence, probabilities }`.
6. The UI shows the label, confidence, a recommended action, and charts. Batch mode adds totals, filters, and CSV export.

The same container runs locally, in Docker, and on Azure. Trained artifacts (`model.joblib`, `vectorizer.joblib`) are baked into the image so Azure does not need to retrain or download Hugging Face data at runtime.

---

## 6. Dataset

### Training data

| Item | Detail |
|------|--------|
| Name | Amazon Polarity |
| Hugging Face id | [`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity) |
| Domain | Amazon product reviews (including food-related products) |
| Original size | About 4 million labelled reviews |
| Subset used | 20,000 balanced reviews (10,000 negative, 10,000 positive) |
| Train / test | 15,999 / 4,000, stratified |
| Labels | `0` = negative, `1` = positive |
| Text fields | `title` and `content` concatenated |

The training script streams the dataset, collects a balanced 20k sample, cleans text, trains the model, and writes artifacts to `models/`.

### Demo data

[`data/sample_reviews.csv`](data/sample_reviews.csv) is a **small hand-written** food-delivery sample used by the UI (“Load sample dataset”) and `GET /sample-data`. It is **not** the training set.

---

## 7. AI and machine-learning approach

### Why this model

Transformer models (for example DistilBERT) can score slightly higher but are heavier to start on a free Azure plan. TF-IDF + Logistic Regression is:

- Fast to train (minutes on a laptop)
- Small enough to load at App Service startup
- Accurate enough for a clear demo (~90% on the test split)
- Easy to explain in an assignment (features, coefficients, metrics)

### Pipeline

| Step | Implementation |
|------|----------------|
| Cleaning | Lowercase; strip HTML tags and URLs; keep letters, digits, apostrophes |
| Features | `TfidfVectorizer`: max 20,000 features, n-grams (1, 2), `min_df=2`, `max_df=0.9`, `sublinear_tf` |
| Classifier | `LogisticRegression`: `liblinear`, `C=2.0`, `max_iter=1000` |
| Persistence | `joblib` files in `models/` |
| Inference | Same `clean_text` function in `app/predict.py` as in training |

### Test results (4,000 reviews)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negative | 0.8985 | 0.8940 | 0.8962 | 2,000 |
| Positive | 0.8945 | 0.8990 | 0.8968 | 2,000 |
| **Accuracy** | | | **0.8965** | 4,000 |

Confusion matrix: **TN 1788**, **FP 212**, **FN 202**, **TP 1798**.

Full JSON metrics: [`models/metrics.json`](models/metrics.json).

### Retraining

```bash
pip install -r requirements-train.txt
python train/train_model.py
```

Then rebuild and redeploy the Docker image so Azure serves the new artifacts.

---

## 8. System architecture

```text
                         Users
                           |
          +----------------+----------------+
          |                                 |
     Signal web UI                    Swagger / curl
     (HTML / CSS / JS)                 /docs, /predict
          |                                 |
          +----------------+----------------+
                           |
                    Azure App Service
                 (Linux container, B1)
                           |
                    FastAPI + Uvicorn
                     port 8000
                           |
              model.joblib + vectorizer.joblib
                           |
                     Docker Hub
        madushansenavirathna/review-sentiment:latest
```

**Design choices**

- One process (FastAPI serves both UI and API) so Docker and Azure stay simple.
- No database: prediction is stateless.
- No GPU and no Azure OpenAI: keeps cost and startup time low.
- Image architecture is **linux/amd64**. An Apple Silicon (ARM) image will not start on App Service.

---

## 9. Application features

### Signal web UI

The home page (`/`) is branded **Signal**.

**Single review**

- Paste a review (up to 8,000 characters)
- Load sample text: Great product / Poor delivery / Helpful support
- See sentiment, confidence, recommended action, and probability bars

**Batch analysis**

- Up to 50 reviews, one per line
- Load the bundled sample CSV
- Summary cards (positive / negative / average confidence)
- Filter rows and export `sentiment-results.csv`

**Dataset validation table**

- Runs the model on `sample_reviews.csv`
- Compares each predicted label with the CSV label
- Shows agreement percentage and matched / mismatched rows
- Can refresh on demand

Recommended actions in the UI are **rules on top of the model** (for example high-confidence negatives suggest follow-up). They are not a second ML model.

### Health and docs

- `/health` — confirms the service is up and the model loaded
- `/docs` — interactive Swagger UI generated by FastAPI

---

## 10. API reference

Base URL (Azure): `https://review-sentiment-msdjshan.azurewebsites.net`  
Base URL (local): `http://127.0.0.1:8000`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Signal UI |
| GET | `/health` | Liveness and model status |
| GET | `/sample-data` | Download demo CSV |
| GET | `/docs` | Swagger |
| POST | `/predict` | One review |
| POST | `/predict/batch` | 1–50 reviews |

### POST `/predict`

**Request**

```json
{
  "text": "The pasta was delicious and arrived hot. Will order again!"
}
```

**Response**

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

`text` must be 1–8,000 characters. Empty text after cleaning returns HTTP 400. Missing model files return HTTP 503.

### POST `/predict/batch`

```json
{
  "texts": [
    "Loved the spicy ramen. Fast delivery.",
    "Wrong order twice. Refund took forever."
  ]
}
```

Returns `{ "results": [ ...PredictResponse ] }` in the same order as the input.

### GET `/health`

```json
{ "status": "ok", "model_loaded": true }
```

More curl examples: [`sample_requests.http`](sample_requests.http).

---

## 11. Technology stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.11 / 3.12 |
| API | FastAPI, Uvicorn, Pydantic |
| ML | scikit-learn, joblib, pandas, numpy |
| Training download | Hugging Face `datasets` |
| Frontend | HTML, CSS, vanilla JavaScript |
| Container | Docker, `python:3.11-slim`, `linux/amd64` |
| Cloud | Azure App Service (Linux, B1, West US 2) |
| Registry | Docker Hub |
| Source control | GitHub (`DonShan/customer-review-sentiment`) |

Runtime packages: [`requirements.txt`](requirements.txt)  
Training extras: [`requirements-train.txt`](requirements-train.txt)

---

## 12. Project structure

```
customer-review-sentiment/
├── README.md                 Assignment-facing summary (11 required sections)
├── DOCUMENTATION.md          This full project document
├── requirements.txt          App / Docker dependencies
├── requirements-train.txt    Extra packages for training
├── Dockerfile                linux/amd64 production image
├── .dockerignore
├── sample_requests.http
├── app/
│   ├── main.py               Routes, UI, lifespan model load
│   ├── predict.py            Clean text, load joblib, infer
│   └── schemas.py            Request / response models
├── static/
│   ├── index.html            Signal UI
│   ├── styles.css
│   ├── live.css
│   └── app.js                Single, batch, and sample-table logic
├── train/
│   └── train_model.py        Download subset, train, save metrics
├── models/
│   ├── model.joblib
│   ├── vectorizer.joblib
│   └── metrics.json
└── data/
    ├── sample_reviews.csv
    └── README.md
```

---

## 13. Local setup

Requires **Python 3.11 or 3.12**.

```bash
cd "customer-review-sentiment"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open http://127.0.0.1:8000/

To retrain before running the app:

```bash
pip install -r requirements-train.txt
python train/train_model.py
```

---

## 14. Docker

The Dockerfile copies the app, static files, model artifacts, and sample CSV. Uvicorn listens on `PORT` (default 8000).

**Important:** build for `linux/amd64`. App Service cannot run an Apple Silicon (ARM) image.

```bash
docker build --platform linux/amd64 -t review-sentiment:latest .
docker run --rm -p 8000:8000 review-sentiment:latest
```

Publish:

```bash
docker login
docker tag review-sentiment:latest madushansenavirathna/review-sentiment:latest
docker push madushansenavirathna/review-sentiment:latest
```

The Hub repository is **private** on the current Docker plan. Azure App Service is configured with registry username and password so it can pull the image.

---

## 15. Azure cloud deployment

### Resources

| Resource | Name | Role |
|----------|------|------|
| Resource group | `rg-review-sentiment` | Groups all resources |
| App Service plan | `plan-review-sentiment-westus2` | Linux **B1**, 1 instance, West US 2 |
| Web app | `review-sentiment-msdjshan` | Hosts the container |
| Image | `madushansenavirathna/review-sentiment:latest` | Application + model |

East US B1 was rejected (subscription VM quota = 0). West US 2 succeeded.

### App settings

| Setting | Purpose |
|---------|---------|
| `WEBSITES_PORT` | `8000` — Azure must route to Uvicorn’s port |
| `DOCKER_REGISTRY_SERVER_URL` | Docker Hub |
| `DOCKER_REGISTRY_SERVER_USERNAME` / `PASSWORD` | Pull the private image |
| `WEBSITES_ENABLE_APP_SERVICE_STORAGE` | `false` |
| `DEPLOY_VERSION` | Changed on each release so Azure pulls `:latest` |

### First-time create

```bash
az login
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
```

### Update after code changes

```bash
docker build --platform linux/amd64 -t madushansenavirathna/review-sentiment:latest .
docker push madushansenavirathna/review-sentiment:latest
az webapp restart --name review-sentiment-msdjshan --resource-group rg-review-sentiment
```

Wait one to two minutes, then hard-refresh the browser.

### Logs

`az webapp log download` can return **504** if the site is starting or the image failed to pull. Prefer:

```bash
az rest --method POST \
  --uri "/subscriptions/<subscription-id>/resourceGroups/rg-review-sentiment/providers/Microsoft.Web/sites/review-sentiment-msdjshan/containerlogs?api-version=2023-12-01"
```

---

## 16. Cost and budget

Almost all Azure spend is the **B1 App Service plan**. The web app itself has no extra licence fee. Docker Hub is free. There is no database, GPU VM, or Azure OpenAI.

| Item | Typical cost |
|------|----------------|
| Linux App Service B1, 24/7 | About **USD 12–15 per month** (~$0.017/hour) |
| Outbound traffic for a student demo | Effectively $0 |
| Docker Hub | $0 |
| Resource group | $0 |

Official prices: [App Service Linux pricing](https://azure.microsoft.com/en-us/pricing/details/app-service/linux/).

The **F1 Free** plan cannot reliably host a custom container, which is why B1 was used.

The Azure free / student credit (often about **$200 for 30 days**) covers development and marking. After the assignment, delete the resource group to stop charges:

```bash
az group delete --name rg-review-sentiment --yes
```

**Stopping the web app** takes the site offline but the **B1 plan can still bill** because the compute is reserved. Use delete (or a budget alert) if you need zero ongoing cost.

---

## 17. Testing and evaluation

| Check | Result |
|-------|--------|
| Held-out test accuracy | 89.65% |
| Local `POST /predict` | Positive and negative samples classify correctly |
| Azure `/health` | `{ "status": "ok", "model_loaded": true }` |
| Azure UI | Signal page, `/static/app.js`, `/sample-data` return 200 after deploy |
| Batch | Up to 50 texts via `/predict/batch` |

Suggested demo script for examiners:

1. Open https://review-sentiment-msdjshan.azurewebsites.net (start the app first if it was stopped).
2. Analyze a positive sample and a negative sample.
3. Open Batch analysis → Load sample dataset → Analyze all reviews.
4. Open `/docs` and call `/predict`.
5. Show GitHub README + this document + the live URL.

---

## 18. Limitations and future work

- **Binary labels only.** Neutral or mixed reviews are forced into positive or negative.
- **English-oriented training data.** Other languages will be weaker.
- **Not a large language model.** Sarcasm and very short slang can be wrong; low confidence is shown so staff can review those cases.
- **No user accounts or stored history.** Predictions are not saved on the server.
- **Private Docker Hub + plan limits.** A public image or Azure Container Registry would simplify pulls.
- **Single B1 instance.** No autoscale; enough for a demo, not a national marketplace.

Possible extensions: multilingual models, aspect-based sentiment (food vs delivery), authentication, Application Insights, and a consumption-based host that scales to zero.

---

## 19. How to start and stop the live site

The App Service may be **stopped** to reduce use when nobody is demoing.

**Start (site becomes available again):**

```bash
az webapp start --name review-sentiment-msdjshan --resource-group rg-review-sentiment
```

Wait 1–2 minutes, then open https://review-sentiment-msdjshan.azurewebsites.net

**Stop:**

```bash
az webapp stop --name review-sentiment-msdjshan --resource-group rg-review-sentiment
```

---

## 20. References

- Assignment: *End-to-End AI Application Development and Cloud Deployment*
- Dataset: [fancyzhx/amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) on Hugging Face
- FastAPI: https://fastapi.tiangolo.com/
- scikit-learn: https://scikit-learn.org/
- Azure App Service: https://learn.microsoft.com/azure/app-service/
- Azure Linux App Service pricing: https://azure.microsoft.com/en-us/pricing/details/app-service/linux/
- Source code: https://github.com/DonShan/customer-review-sentiment
- Live app: https://review-sentiment-msdjshan.azurewebsites.net
