from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.predict import is_loaded, load_artifacts, predict_many, predict_one
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
SAMPLE_DATA = ROOT / "data" / "sample_reviews.csv"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Customer Review Sentiment API",
    description=(
        "Classify e-commerce and food-delivery customer reviews as positive or negative. "
        "Use the web UI at `/` or the JSON API at `/predict`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web UI not found.")
    return FileResponse(index_path)


@app.get("/sample-data", include_in_schema=False)
def sample_data():
    if not SAMPLE_DATA.exists():
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
    return FileResponse(SAMPLE_DATA, media_type="text/csv", filename="sample_reviews.csv")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=is_loaded())


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = predict_one(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    try:
        results = predict_many(payload.texts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return BatchPredictResponse(results=results)
