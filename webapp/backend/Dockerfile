FROM python:3.11-slim

WORKDIR /app

# Force HuggingFace Hub into offline mode so it never tries to validate
# local absolute paths as Hub repo IDs (huggingface_hub >= 0.20 regression).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Install system deps (needed by some tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
