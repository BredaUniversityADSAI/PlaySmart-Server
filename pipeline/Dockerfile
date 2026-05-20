FROM python:3.10-slim
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        cron tzdata procps \
    && rm -rf /var/lib/apt/lists/*

# Python deps 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Cron setup 
COPY crontab /etc/cron.d/pipeline-cron
RUN chmod 0644 /etc/cron.d/pipeline-cron

CMD ["sh", "-c", "echo 'Pipeline container ready' && cron -f"]
