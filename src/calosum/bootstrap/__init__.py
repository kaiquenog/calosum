"""
# Bootstrap Boundary
Ponto principal de injeção de dependências (CLI, Factory, Settings).
**Invariante de Design:** Configura a relação entre `domain` e `adapters` no startup, mas nunca interfere no ciclo de vida (event loop interno) do sistema.
"""
