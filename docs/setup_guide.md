# Setup Guide

## Prerequisites

- Python 3.8+
- PostgreSQL
- Qdrant vector database

## Installation

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and configure
6. Run ingestion pipeline: `python scripts/run_ingestion_pipeline.py`

## Running

Start API: `uvicorn api.main:app --reload`
Start UI: `streamlit run ui/streamlit_app.py`


