# Deployment

This repo includes a `Dockerfile` and `docker-compose.yml`.

## Docker Compose (recommended)

```bash
docker-compose up --build
```

## Docker (single container)

```bash
docker build -t ask-rag .
docker run --rm -p 8501:8501 --env-file .env ask-rag
```
