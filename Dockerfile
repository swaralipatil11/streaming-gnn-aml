FROM python:3.10-slim

# Install system dependencies (native compilation tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies without caching.
# We first install CPU-only torch to prevent large CUDA layers (keeps container size minimal).
# Then install remaining packages from requirements.txt.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
