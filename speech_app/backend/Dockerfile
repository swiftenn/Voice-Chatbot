# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system-level dependencies (audio + whisper-related)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libespeak1 \
    libportaudio2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install heavy libraries (whisper + torch) separately for better caching
RUN pip install --no-cache-dir  --timeout=600 torch openai-whisper

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose the app port
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
