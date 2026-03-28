FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md AGENTS.md /app/
COPY docs /app/docs
COPY src /app/src
COPY examples /app/examples

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["python3", "-m", "calosum.bootstrap.api"]
