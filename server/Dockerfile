FROM python:3.10-slim

# Set working directory
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY server/requirements.txt ./server/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r server/requirements.txt

# Copy source code
COPY src/ ./src/
COPY server/ ./server/

# Create directories for results
RUN mkdir -p /app/result

# Create non-root user with uid 1000 to match host user
RUN useradd -m -u 1000 raguser && \
    chown -R raguser:raguser /app

# Switch to non-root user
USER raguser

# Expose port
EXPOSE 3003

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:3003/health')"

# Start RAG server
CMD ["python3", "server/rag_server.py"]

