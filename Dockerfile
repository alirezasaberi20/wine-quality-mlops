# Wine Quality Classification - Multi-stage Dockerfile
# Stage 1: Builder - Install dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only production dependencies (exclude dev/test packages)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    pandas==2.1.4 \
    numpy==1.26.2 \
    scikit-learn==1.3.2 \
    fastapi==0.108.0 \
    uvicorn==0.25.0 \
    pydantic==2.5.3 \
    pyyaml==6.0.1 \
    requests==2.31.0 \
    streamlit==1.29.0

# Stage 2: Production image
FROM python:3.11-slim as production

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY streamlit_app/ ./streamlit_app/
COPY models/ ./models/
COPY config.yaml .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Default command - run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative commands:
# For Streamlit: streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0
