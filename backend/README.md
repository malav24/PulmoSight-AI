# Backend (Stage 2)

FastAPI app wrapping the Stage-1 ML modules. Run from the project root:

    pip install -r backend/requirements.txt
    uvicorn backend.app.main:app --reload --port 8000

Docs: ../docs/API_DOCUMENTATION.md - tests: `python -m pytest backend/tests`
