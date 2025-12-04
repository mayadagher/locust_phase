# Simple Python base image
FROM python:3.12-slim

# Install system packages needed for compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libhdf5-dev \
    libgl1 \
    libglib2.0-0 \
    libblas-dev \
    liblapack-dev \
    libsuitesparse-dev \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set default command
CMD ["python", "-u", "/app/src/main.py"] # -u for unbuffered output
