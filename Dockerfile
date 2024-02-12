FROM python:3.9-slim-buster

WORKDIR /app

COPY ./ /app

RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip  -r requirements.txt
