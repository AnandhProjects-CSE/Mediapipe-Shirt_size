FROM python:3.11.9-slim

ENV PYTHONUNBUFFERED=1

# System libraries required by OpenCV & MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", "--workers", "1", "--threads", "1", "--timeout", "300"]
