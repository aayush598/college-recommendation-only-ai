# Use official Python 3.12 slim image
FROM python:3.12-slim

# Prevent Python from writing pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (optional, only if needed for your libs)
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command to run the app with Uvicorn (FastAPI)
# Replace 'app:app' with 'yourfilename:app' if the FastAPI instance is in another file
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
