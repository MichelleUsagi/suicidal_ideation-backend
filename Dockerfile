# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all app source code
COPY . .

# Ensure model file exists (for fail-fast behavior)
RUN test -f ./models/suicide_detector_model.h5 || (echo "❌ Model file not found!" && exit 1)

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start FastAPI using uvicorn on port 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

COPY ./models/suicide_detector_model.h5 /app/models/suicide_detector_model.h5
COPY ./models/tokenizer.json /app/models/tokenizer.json
