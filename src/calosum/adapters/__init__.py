"""
# Adapters Boundary
Armazena a sujeira do mundo real e as bibliotecas (HTTPX, Qdrant, SDKs).
**Invariante de Design:** Adaptadores respeitam os contratos do `shared.ports` à risca. Não tomam decisões metacognitivas ou arbitram regras; apenas traduzem intenções.
"""
