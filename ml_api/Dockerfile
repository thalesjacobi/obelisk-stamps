# ML API Service Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV (required by ultralytics)
# Note: libgl1 replaces libgl1-mesa-glx in newer Debian versions
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY ml_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML utilities (needed for custom Keras layers)
COPY ml/model_utils.py ml/model_utils.py
COPY ml/__init__.py ml/__init__.py

# Copy API code
COPY ml_api/app.py .

# Expose port
EXPOSE 8081

# Run with gunicorn for production
# --timeout 300: Allow long requests (model inference can be slow)
# --graceful-timeout 300: Allow graceful shutdown time
# --preload: Load app before forking workers (shares memory)
CMD ["gunicorn", "--bind", "0.0.0.0:8081", "--workers", "1", "--threads", "4", "--timeout", "300", "--graceful-timeout", "300", "--preload", "app:app"]
