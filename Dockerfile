FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy package metadata first for better caching
COPY pyproject.toml README.md LICENSE.md ./
COPY src ./src

# Install dependencies
RUN pip install --no-cache-dir .

# Expose the hardcoded port
EXPOSE 7832

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7832/health || exit 1

# Run the server
CMD ["biel-mcp"]
