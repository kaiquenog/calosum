# AGENTS.md

Este arquivo e um mapa curto do repositorio. Nao e a documentacao completa.

## Comece Aqui

- Leia [docs/index.md](docs/index.md) para navegar o conhecimento versionado do projeto.
- Leia [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) antes de mudar dependencias, camadas ou pontos de integracao.
- Leia [docs/PLANS.md](docs/PLANS.md) antes de iniciar mudancas maiores que um arquivo, mudancas cross-cutting ou qualquer alteracao de arquitetura.
- Consulte [docs/QUALITY_SCORE.md](docs/QUALITY_SCORE.md) e [docs/RELIABILITY.md](docs/RELIABILITY.md) ao tocar confiabilidade, runtime, memoria, telemetria ou operacao.
- Leia [docs/INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md) ao tocar bootstrap, perfis ou `docker-compose`.

## Commands

### Python Backend

```bash
PYTHONPATH=src ./.venv/bin/python3 -m unittest discover -s tests -t . # All tests
PYTHONPATH=src ./.venv/bin/python3 -m unittest tests/bootstrap/test_factory.py # Single file
PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks # Governance (or `calosum-harness`)
PYTHONPATH=src ./.venv/bin/python3 -m calosum.bootstrap.cli chat # CLI (or `calosum`)
PYTHONPATH=src ./.venv/bin/python3 -m calosum.bootstrap.api # API Server
```

### TypeScript UI

```bash
cd ui
npm install # Dependencies
npm run dev # Dev server
npm run build # Prod build
npm run lint # Lint
```

### Docker
```bash
docker compose -f deploy/docker-compose.yml up -d # Start
docker compose -f deploy/docker-compose.yml ps # Health
```

### Pre-Commit Checklist

1. `PYTHONPATH=src ./.venv/bin/python3 -m calosum.harness_checks` — governance checks
2. `PYTHONPATH=src ./.venv/bin/python3 -m unittest discover -s tests -t .` — all tests pass
3. For UI changes: `cd ui && npm run lint && npm run build`

## Code Style

### Python

- **Python version:** 3.11+
- **Imports:** `from __future__ import annotations` at top of every file. Group imports: stdlib → third-party → internal (`calosum.*`). Sort alphabetically within groups.
- **Typing:** Use type annotations on all function signatures and class attributes. Prefer `X | None` over `Optional[X]`. Use `Protocol` for interfaces, not ABCs.
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants. Portuguese-English mix is accepted per existing convention.
- **Dataclasses:** Prefer `@dataclass` for value objects. Use `field(default_factory=...)` for mutable defaults.
- **Async:** Provide both sync and async variants when exposing ports (e.g., `reason` / `areason`). Use `asyncio` for concurrent orchestration.
- **Error handling:** Never crash on optional infrastructure failure. Degrade gracefully with explicit fallbacks. Log warnings, don't raise.
- **Docstrings:** Required for public classes and methods. Portuguese preferred for domain concepts.
- **Module size:** Hard limit of 400 lines per module, enforced by `harness_checks.py`.
- **No SDK imports in domain:** The `domain/` layer must never import external SDKs (torch, transformers, httpx, etc.). All external deps live in `adapters/`.

### TypeScript

- **Framework:** React 19 with Vite, Tailwind CSS 4, TypeScript ~5.9
- **Linting:** ESLint 9 with `typescript-eslint`, `react-hooks`, `react-refresh` plugins
- **Typing:** Strict TypeScript. No `any` (enforced by ESLint rule).
- **Naming:** `camelCase` for variables/functions, `PascalCase` for components, `kebab-case` for files.
- **Components:** Functional components only. Use hooks for state. No class components.
- **Build:** `tsc -b && vite build` — type check must pass before build.

## Architecture Rules

### Layer Boundaries (Ports and Adapters)

1. **`shared/`** — No internal dependencies. Pure types, Protocols, descriptors, serialization.
2. **`domain/`** — Core business logic. NEVER imports from `adapters/` or `bootstrap/`. Only imports `shared/` and sibling domain modules.
3. **`adapters/`** — Concrete implementations of Protocols. All external SDKs live here. Obeys `shared/` interfaces blindly.
4. **`bootstrap/`** — Only layer authorized to wire adapters into domain. Contains CLI, API, settings, factory.

### Key Constraints

- New external integrations must go behind a `Protocol` in `shared/ports.py`.
- Register new modules in `MODULE_RULES` inside `harness_checks.py` with allowed internal imports.
- Missing module rules break the build with `missing_module_rule`.
- Cross-cutting changes require a versioned plan in `docs/exec-plans/active/`.
- Move completed plans to `docs/exec-plans/completed/` with a summary.

## Working Rules

- Keep important knowledge inside the repo. External conversations don't count as source of truth.
- When a recurring rule emerges, promote it to mechanical checking in `harness_checks.py`.
- Register structural debt in `docs/exec-plans/tech-debt-tracker.md`.
- The bootstrap must prefer explicit fallback over hard failure when optional infrastructure is unavailable.
- The agent entry point is `orchestrator.py`. CLI (`cli.py`) and API (`api.py`) must not leak context into `domain/`.

## Atalhos

- Arquitetura: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Planos: [docs/PLANS.md](docs/PLANS.md)
- Qualidade: [docs/QUALITY_SCORE.md](docs/QUALITY_SCORE.md)
- Confiabilidade: [docs/RELIABILITY.md](docs/RELIABILITY.md)
- Infraestrutura: [docs/INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md)
- Roadmap de producao: [docs/production-roadmap.md](docs/production-roadmap.md)
- Referencias de harness engineering: [docs/references/harness-engineering.md](docs/references/harness-engineering.md)
