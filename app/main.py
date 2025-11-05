# app/main.py
from fastapi import FastAPI
from app.db import engine
from app.models import Base  # ensures models are imported before create_all
from app.routes import router

app = FastAPI()

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

app.include_router(router)
