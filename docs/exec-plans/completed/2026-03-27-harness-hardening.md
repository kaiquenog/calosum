# Harness Hardening For Project Growth

## Purpose

Aplicar principios de harness engineering para reduzir drift arquitetural e tornar crescimento do projeto mecanicamente verificavel.

## Scope

- criar mapa curto para agentes
- estruturar docs como sistema de registro
- formalizar planos e debt tracker
- adicionar harness checks para docs e arquitetura
- cobrir isso com testes

## Validation

- `PYTHONPATH=src python3 -m calosum.harness_checks`
- `PYTHONPATH=src python3 -m unittest discover -s tests -t .`

## Progress

- concluido em 2026-03-27

## Decision Log

- adotado `AGENTS.md` curto em vez de um manual gigante
- adotadas checagens AST para fronteiras arquiteturais
- adotado tracker de debt versionado no repositorio
