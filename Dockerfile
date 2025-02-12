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
RUN pip install --no-cache-dir --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the application code
COPY . .

# Stage 2: Final
FROM python:3.11-slim AS final

WORKDIR /app

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

# Ensure scripts in .local are usable (for pip installations that might land in .local)
ENV PATH=/root/.local/bin:$PATH

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]