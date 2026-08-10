FROM ghcr.io/railwayapp/railpack-runtime:mise-2026.7.15

WORKDIR /app

# Copy Python version config
COPY runtime.txt .

# Install Python using mise
RUN mkdir -p /etc/mise && \
    curl -fsSL https://mise.jdx.dev/install.sh | sh && \
    /root/.local/bin/mise install python@3.11.11 && \
    /root/.local/bin/mise exec -- python -m venv /app/.venv

# Copy requirements and install dependencies
COPY requirements.txt .
RUN /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download model at build time
RUN /app/.venv/bin/python -c "
import os
import urllib.request

MODEL_PATH = '/app/model.keras'
MODEL_URL = 'https://media.githubusercontent.com/media/immsoumya7676-web/packscan/main/model.keras'

print(f'Downloading model from {MODEL_URL}...')
try:
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    size = os.path.getsize(MODEL_PATH)
    print(f'✓ Model downloaded successfully ({size} bytes)')
except Exception as e:
    print(f'⚠ Model download failed: {e}')
    print('  App will attempt to download at startup')
"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

