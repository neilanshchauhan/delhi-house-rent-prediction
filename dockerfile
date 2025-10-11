FROM python:3.11-slim
WORKDIR /app

# Copy requirements first for better caching
COPY ./src/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./src /app/src
COPY ./data /app/data
COPY ./models /app/models

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]