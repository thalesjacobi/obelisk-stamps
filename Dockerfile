# Use a Debian-based Python image
#FROM python:3.11-slim
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install required packages
#RUN apt-get update && \
#    apt-get install -y gcc libpq-dev python3-dev musl-dev && \
#    rm -rf /var/lib/apt/lists/* 

# Upgrade pip and install Python dependencies with detailed logging
RUN echo "=== Step 1: Upgrading pip ===" && \
    pip install --upgrade pip && \
    echo "✓ Pip upgraded successfully" && \
    echo "" && \
    echo "=== Step 2: Installing Python dependencies from requirements.txt ===" && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "✓ All dependencies installed successfully" && \
    echo "" && \
    echo "=== Step 3: Verifying installed packages ===" && \
    pip list && \
    echo "" && \
    echo "=== Step 4: Checking pip compatibility ===" && \
    pip check && \
    echo "✓ All package dependencies are compatible" && \
    echo "" && \
    echo "=== Docker build setup complete ==="

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 8080

# Run the application using python directly (reads PORT from environment)
CMD ["python", "app.py"]