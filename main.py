import os
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.core.config import settings
from src.api.routes import router as api_router

app = FastAPI(
    title="Autonomous Data Analyst API",
    description="Backend API for the LangGraph-powered Autonomous Data Analyst",
    version="1.0.0"
)

# CORS config to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Update with specific frontend domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"status": "online", "message": "Autonomous Data Analyst Backend is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)
