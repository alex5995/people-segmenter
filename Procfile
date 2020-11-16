release: python download.py
web: uvicorn segmenter.main:app --host=0.0.0.0 --port=${PORT:-5000}