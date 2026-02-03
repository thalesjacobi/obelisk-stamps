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

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 8080

# Run the application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]