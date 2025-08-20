FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY biel_mcp_server.py .

# Expose the hardcoded port
EXPOSE 7832

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7832/sse || exit 1

# Run the server
CMD ["python", "biel_mcp_server.py"] 