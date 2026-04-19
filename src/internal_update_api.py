import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.update_master import run_update_master

app = FastAPI(title="Baseball Predict Internal API")


class UpdateResponse(BaseModel):
    status: str
    message: str
    master_file: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app = FastAPI(title="Baseball Predict Internal API")


class UpdateResponse(BaseModel):
    status: str
    message: str
    master_file: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/internal/run-update", response_model=UpdateResponse)
def run_update(x_update_token: str | None = Header(default=None)) -> UpdateResponse:
    expected_token = os.getenv("INTERNAL_UPDATE_TOKEN")

    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="INTERNAL_UPDATE_TOKEN is not configured",
        )

    if x_update_token != expected_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    master_file = run_update_master()

    return UpdateResponse(
        status="ok",
        message="Master actualizado correctamente",
        master_file=str(master_file),
    )