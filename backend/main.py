from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.upload import router as upload_router
from .routes.chat import router as chat_router

_VERSION = "2.0.0"

app = FastAPI(
    title="DataLens AI",
    version=_VERSION,
    description="Autonomous Data Analysis Platform — FastAPI backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": _VERSION}
