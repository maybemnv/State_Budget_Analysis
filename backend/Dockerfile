# Use a slim Python image for the base
FROM python:3.13-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for prophet, pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# --frozen ensures we use the exact versions from uv.lock
RUN uv sync --frozen --no-install-project --no-dev

# Second stage: final image
FROM python:3.13-slim

WORKDIR /app

# Copy the virtual environment from the base stage
COPY --from=base /app/.venv /app/.venv

# Add the virtual environment's bin to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy the backend source code
COPY backend/ ./backend/
# Copy requirements/config if they are in root and needed (usually backend/ contains its own logic)
# Based on the structure, backend/ is a subfolder. 
# We'll run from the root of /app, so 'backend.main:app'

# Expose the port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
