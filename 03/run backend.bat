@echo off
REM Move to the directory that contains the backend package
cd /d "C:\Users\Laptop\Desktop\Coding\LLM Engineering Projects\RAG\03"

REM Activate conda / venv if needed (optional)
REM call C:\Users\Laptop\anaconda3\Scripts\activate.bat base

REM Run FastAPI via uvicorn as a module
uvicorn backend.main:app --reload

pause
