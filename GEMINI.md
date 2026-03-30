# Calosum - Project Context for Gemini

## Project Overview

**Calosum** is a neuro-symbolic AI agent framework featuring a dual-hemisphere cognitive architecture. It combines perception based on embeddings (Right Hemisphere) with LLM-based reasoning (Left Hemisphere), safe action execution, and metacognition inspired by Group-Evolving Agents (GEA).

### Key Technologies
*   **Backend:** Python 3.11+, FastAPI, Qdrant, Transformers, PyTorch, DSPy.
*   **Frontend (Telemetry UI):** React 19, TypeScript, Vite, Tailwind CSS 4.
*   **Infrastructure:** Docker Compose (Qdrant, OpenTelemetry Collector, Jaeger).

### Architecture
The project strictly adheres to the **Ports and Adapters (Hexagonal)** architectural pattern.
*   `src/calosum/shared/`: Pure interfaces (Protocols), types, and utilities. **Zero dependencies**.
*   `src/calosum/domain/`: Core cognitive logic. **Never imports from `adapters/` or `bootstrap/`**. No external SDKs (like `torch`, `openai`, `transformers`) are permitted here.
*   `src/calosum/adapters/`: Concrete implementations of Ports. All external API calls and ML specific code lives here.
*   `src/calosum/bootstrap/`: Dependency injection and application entrypoints (CLI, API).

This architectural integrity is statically enforced by AST checks.

## Building and Running

### Backend (Python)

*   **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
*   **Run Interactive Chat:**
    ```bash
    python3 -m calosum.bootstrap.cli chat
    ```
*   **Run API Server:**
    ```bash
    python3 -m calosum.bootstrap.api
    ```
*   **Run Tests:**
    ```bash
    PYTHONPATH=src python3 -m unittest discover -s tests -t .
    ```
*   **Check Architectural Integrity (Crucial before committing):**
    ```bash
    PYTHONPATH=src python3 -m calosum.harness_checks
    ```

### Frontend (React UI)

*   **Install & Run Development Server:**
    ```bash
    cd ui && npm install && npm run dev
    ```

### Infrastructure (Docker)

*   **Start supporting services (Qdrant, OTLP, Jaeger):**
    ```bash
    docker compose -f deploy/docker-compose.yml up --build -d
    ```

## Development Conventions

1.  **Strict Layering:** You must respect the Ports and Adapters boundaries. Violations will break the build (`harness_checks.py`).
2.  **No External SDKs in Domain:** Do not use libraries like `torch`, `transformers`, `httpx`, or `openai` within `src/calosum/domain/` or `src/calosum/shared/`.
3.  **Adding Integrations:** To add a new integration, define a `Protocol` in `shared/ports.py`, implement it in `adapters/`, and wire it up in `bootstrap/factory.py`.
4.  **File Size Limit:** There is a strict limit of 400 lines per file.
5.  **Planning Requirements:** Changes affecting more than one subsystem require a documented plan in `docs/exec-plans/active/`.
