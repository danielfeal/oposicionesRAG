# Chainlit app image. Runs `chainlit run app.py`, talking to the `qdrant`
# and `ollama` services over the compose network (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py chainlit.md ./
COPY .chainlit/config.toml .chainlit/config.toml
COPY .chainlit/translations/es.json .chainlit/translations/es.json
COPY rag/ rag/
COPY scripts/ scripts/

EXPOSE 8000

# -h (headless): no browser to open inside a container.
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000", "-h"]
