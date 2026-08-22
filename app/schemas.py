from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000, description="Customer review text")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)


class ProbabilityScores(BaseModel):
    negative: float
    positive: float


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: ProbabilityScores


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
