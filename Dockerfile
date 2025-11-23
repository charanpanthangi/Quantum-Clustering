# Use a slim Python image to keep the container lightweight.
FROM python:3.11-slim

# Install system dependencies required by scientific Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container.
WORKDIR /app

# Copy dependency list first to leverage Docker layer caching.
COPY requirements.txt requirements.txt

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository contents.
COPY . .

# Default command runs the demo on the moons dataset using angle encoding.
CMD ["python", "app/main.py", "--dataset", "moons", "--clusters", "2", "--iters", "10", "--feature-map", "angle"]
