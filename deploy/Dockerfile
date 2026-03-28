FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY examples /app/examples

ENV PYTHONPATH=/app/src

CMD ["python3", "examples/cognitive_cycle.py"]
