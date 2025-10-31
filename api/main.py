"""FastAPI application"""
from fastapi import FastAPI
from api.routes import query, documents, health

app = FastAPI(title="AI Policy Assistant API")

app.include_router(query.router)
app.include_router(documents.router)
app.include_router(health.router)

@app.get("/")
async def root():
    return {"message": "AI Policy Assistant API"}






