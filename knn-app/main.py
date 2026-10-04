from math import isfinite
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from KNN.classifieddata1 import (
    FEATURES,
    DEFAULT_NEIGHBORS,
    evaluate_model,
    get_dataset_summary,
    predict_one,
)


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="KNN Studio", description="A small K-nearest-neighbors learning app")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


class PredictionInput(BaseModel):
    WTT: float = Field(allow_inf_nan=False)
    PTI: float = Field(allow_inf_nan=False)
    EQW: float = Field(allow_inf_nan=False)
    SBI: float = Field(allow_inf_nan=False)
    LQE: float = Field(allow_inf_nan=False)
    QWG: float = Field(allow_inf_nan=False)
    FDJ: float = Field(allow_inf_nan=False)
    PJF: float = Field(allow_inf_nan=False)
    HQE: float = Field(allow_inf_nan=False)
    NXJ: float = Field(allow_inf_nan=False)


def page_context(request: Request, active_page: str, **values: object) -> dict[str, object]:
    return {"request": request, "active_page": active_page, **values}


@app.get("/", response_class=HTMLResponse, name="home")
def home(request: Request):
    summary = get_dataset_summary()
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context=page_context(request, "home", summary=summary),
    )


@app.get("/predict", response_class=HTMLResponse, name="predict_page")
def predict_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context=page_context(
            request,
            "predict",
            features=FEATURES,
            values={feature: "" for feature in FEATURES},
            errors={},
            result=None,
        ),
    )


@app.post("/predict", response_class=HTMLResponse, name="predict_form")
def predict_form(
    request: Request,
    WTT: Annotated[str, Form()] = "",
    PTI: Annotated[str, Form()] = "",
    EQW: Annotated[str, Form()] = "",
    SBI: Annotated[str, Form()] = "",
    LQE: Annotated[str, Form()] = "",
    QWG: Annotated[str, Form()] = "",
    FDJ: Annotated[str, Form()] = "",
    PJF: Annotated[str, Form()] = "",
    HQE: Annotated[str, Form()] = "",
    NXJ: Annotated[str, Form()] = "",
):
    values = dict(zip(FEATURES, (WTT, PTI, EQW, SBI, LQE, QWG, FDJ, PJF, HQE, NXJ)))
    parsed: dict[str, float] = {}
    errors: dict[str, str] = {}
    for feature, raw_value in values.items():
        try:
            number = float(raw_value)
            if not isfinite(number):
                raise ValueError
            parsed[feature] = number
        except ValueError:
            errors[feature] = "Enter a finite number."

    result = predict_one(parsed) if not errors else None
    status_code = 200 if result else 422
    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context=page_context(
            request,
            "predict",
            features=FEATURES,
            values=values,
            errors=errors,
            result=result,
        ),
        status_code=status_code,
    )


@app.get("/evaluation", response_class=HTMLResponse, name="evaluation")
def evaluation_page(request: Request):
    evaluation = evaluate_model()
    return templates.TemplateResponse(
        request=request,
        name="evaluation.html",
        context=page_context(
            request,
            "evaluation",
            evaluation=evaluation,
            default_neighbors=DEFAULT_NEIGHBORS,
        ),
    )


@app.get("/health", name="health")
def health():
    return {"status": "ok"}


@app.post("/api/predict", name="api_predict")
def api_predict(payload: PredictionInput):
    return predict_one(payload.model_dump())