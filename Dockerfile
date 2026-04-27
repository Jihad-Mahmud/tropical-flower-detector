# Use slim Python image
FROM python:3.11-slim
 
WORKDIR /app
 
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*
 
# Install CPU-only PyTorch first (much smaller — 800MB instead of 5GB)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
 
# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy project files
COPY . .
 
# Run the server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
