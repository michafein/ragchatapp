# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies (including mupdf and freetype for PyMuPDF)
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    gcc \
    libmupdf-dev \
    libfreetype6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies without caching to reduce image size
RUN pip install --no-cache-dir --user -r requirements.txt 

# Copy the application code
COPY . .

# Stage 2: Final
FROM python:3.11-slim AS final

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app_enhanced.py \
    PATH=/root/.local/bin:$PATH

# Install runtime dependencies for PyMuPDF (no need for extra packages like CUDA)
RUN apt-get update && apt-get install --no-install-recommends -y \
    libmupdf-dev \
    libfreetype6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /root/.local /root/.local

# Copy the application code from the builder stage
COPY --from=builder /app /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads /app/embeddings /app/pages_and_chunks /app/logs && \
    chmod -R 755 /app

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Healthcheck to verify app is running correctly
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# Run the enhanced application
CMD ["python", "app.py"]