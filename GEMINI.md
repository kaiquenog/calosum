# GEMINI Context: Calosum Deployment

This directory contains the infrastructure and deployment configurations for the Calosum project, focusing on containerized orchestration, observability, and data persistence.

## Project Overview

Calosum is a Python-based system (likely an AI/LLM orchestrator) designed for modularity and deep observability. The deployment stack uses Docker Compose to orchestrate several key components:

- **Orchestrator**: The core Python application (`calosum.bootstrap.entry.api`) running on Python 3.13.
- **Qdrant**: A vector database used for semantic search and high-dimensional data storage.
- **OpenTelemetry (OTEL)**: A collector that aggregates traces, metrics, and logs.
- **Jaeger**: A distributed tracing backend for visualizing application flow and performance.

## Core Stack & Technologies

- **Language**: Python 3.13 (Slim image)
- **Containerization**: Docker & Docker Compose
- **Observability**: OpenTelemetry, Jaeger
- **Database**: Qdrant (Vector DB), DuckDB (Local semantic state)
- **Networking**: REST API on port 8000 (Orchestrator), OTLP on 4317/4318.

## Building and Running

### Prerequisites
- Docker and Docker Compose installed.
- A `.env` file in the parent directory (`..`) containing required environment variables (e.g., API keys).

### Key Commands
- **Start the entire stack**:
  ```bash
  docker-compose up
  ```
- **Build the orchestrator image**:
  ```bash
  docker-compose build orchestrator
  ```
- **View logs**:
  ```bash
  docker-compose logs -f orchestrator
  ```

## Key Files & Infrastructure

- **`docker-compose.yml`**: The main orchestration file. It mounts local source code (`../src`) and configuration files into the containers for a seamless development experience.
- **`Dockerfile`**: Builds the `orchestrator` service. It installs the project in editable-like mode via `pip install .` using the `pyproject.toml` found in the root.
- **`otel-collector-config.yaml`**: Configures the OpenTelemetry pipeline. It receives data via OTLP (gRPC/HTTP) and exports traces to Jaeger and metrics/logs to the debug console.

## Development Conventions

- **Observability First**: The system is pre-configured with OpenTelemetry. Ensure new modules use the established tracing and logging patterns.
- **Environment Driven**: Configuration is managed via environment variables (prefixed with `CALOSUM_`) and a `.env` file.
- **Volume Mapping**: The `orchestrator` service maps `../src` to `/app/src`, allowing for hot-reloading or immediate testing of source changes within the container.
