FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3-tk \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install -r requirements-dev.txt \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["python", "-c", "import pymunk, torch, scipy, numpy; print('Swarmball environment is ready')"]