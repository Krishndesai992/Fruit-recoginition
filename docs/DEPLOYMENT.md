# Deployment Guide: Fruit Recognition Web Application

This document outlines deployment strategies and operational best practices for deploying the **Fruit Recognition using Deep Features & SVM** application into production environments.

---

## 1. System Requirements & Architecture

- **Runtime:** Python 3.9+ (Python 3.10 – 3.14 compatible)
- **Memory (RAM):** 2 GB minimum (4 GB recommended for concurrent inference)
- **CPU:** 2+ vCPUs recommended
- **GPU (Optional):** NVIDIA GPU with CUDA support for accelerated feature extraction (CPU is supported by default)
- **Storage:** ~500 MB disk space for dependencies and model weights

---

## 2. Environment Configuration

Copy the example environment file and configure any overrides:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|:---|:---:|:---|
| `FLASK_APP` | `app.py` | Flask entrypoint module |
| `FLASK_ENV` | `production` | Application execution environment (`production` or `development`) |
| `FLASK_DEBUG` | `0` | Debug mode toggle (`0` in production) |
| `FLASK_HOST` | `0.0.0.0` | Network binding interface |
| `FLASK_PORT` | `5000` | Port number to bind HTTP server |
| `SECRET_KEY` | - | Session cryptographic secret key |
| `MODEL_PATH` | `models/svm_model.pkl` | Path to serialized SVM bundle |
| `DEFAULT_BACKBONE` | `resnet50` | Default CNN backbone for feature extraction |

---

## 3. Local & Development Execution

```bash
# 1. Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# 2. Run Flask server
python app.py
```
The application will be accessible at `http://127.0.0.1:5000`.

---

## 4. Production WSGI Deployment

Flask's built-in server (`app.run()`) is single-threaded and not intended for production traffic. Use a production WSGI server.

### Option A: Waitress (Recommended for Windows & Cross-Platform)

1. Install Waitress:
   ```bash
   pip install waitress
   ```

2. Run with multi-threaded workers:
   ```bash
   waitress-serve --host=0.0.0.0 --port=5000 --threads=4 app:app
   ```

### Option B: Gunicorn (Recommended for Linux/macOS)

1. Install Gunicorn:
   ```bash
   pip install gunicorn
   ```

2. Run with 2-4 workers (with timeout allowance for CNN feature extraction):
   ```bash
   gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 2 --timeout 120 app:app
   ```

---

## 5. Docker Containerization

### Dockerfile

A production-grade multi-stage Dockerfile can be structured as follows:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt waitress

# Copy application source code and models
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY models/ ./models/
COPY results/ ./results/
COPY app.py .
COPY .env.example .env

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:5000/api/samples || exit 1

CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--threads=4", "app:app"]
```

### Building and Running the Image

```bash
# Build Docker image
docker build -t fruit-recognition-app:latest .

# Run container
docker run -d -p 5000:5000 --name fruit-vision fruit-recognition-app:latest
```

---

## 6. Cloud Platform Deployments

### Render / Railway / Fly.io

1. Connect your GitHub repository to Render or Railway.
2. Set Environment to **Python 3**.
3. **Build Command:**
   ```bash
   pip install -r requirements.txt waitress
   ```
4. **Start Command:**
   ```bash
   waitress-serve --host=0.0.0.0 --port=$PORT app:app
   ```

### Hugging Face Spaces (Docker or Gradio/Flask)

1. Create a new Space with the **Docker** SDK.
2. Push repository code with the provided `Dockerfile`.
3. Set Space port to `5000` via Space Settings.

---

## 7. Health Check and Verification

After deployment, verify the service is running and healthy:

```bash
# 1. Check UI availability
curl -I http://localhost:5000/

# 2. Check catalog and pipeline health
curl http://localhost:5000/api/samples

# 3. Check performance metrics
curl http://localhost:5000/api/metrics
```
