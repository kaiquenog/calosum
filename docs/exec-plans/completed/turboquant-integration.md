# TurboQuant Integration

## Purpose

Integrar os algoritmos **PolarQuant** e **QJL** (Google Research, ICLR 2026) ao framework
Calosum para comprimir vetores de alta dimensão usados no Hemisfério Direito e no índice do
Qdrant, reduzindo footprint de memória episódica em até 8× e acelerando o cálculo de
similaridade coseno sem perda de acurácia.

Referências: [TurboQuant](https://arxiv.org/abs/2504.19874) ·
[PolarQuant](https://arxiv.org/abs/2502.02617) ·
[QJL](https://arxiv.org/abs/2406.03482)

---

## Scope

### Arquivos Novos

| Arquivo | Camada | Descrição |
|---|---|---|
| `src/calosum/adapters/quantized_embeddings.py` | adapters | `TurboQuantVectorCodec` — PolarQuant encoder + QJL residual + inner product aproximado |
| `src/calosum/adapters/memory_qdrant_serializers.py` | adapters | Helpers de serialização extraídos de `memory_qdrant.py` — `_episode_payload`, `_episode_from_point`, etc. |
| `tests/test_quantized_embeddings.py` | tests | Cobertura de unidade e propriedade do codec |

### Arquivos Modificados

| Arquivo | Mudança |
|---|---|
| `src/calosum/shared/ports.py` | Novo `VectorCodecPort` Protocol |
| `src/calosum/adapters/text_embeddings.py` | Hook opcional `codec: VectorCodecPort \| None` para comprimir output |
| `src/calosum/adapters/memory_qdrant.py` | Extrair serializers para módulo separado (debt); ativar `Scalar Quantization`; usar codec se configurado |
| `src/calosum/adapters/right_hemisphere_hf.py` | `_calculate_surprise()` pode usar inner product aproximado do codec |
| `src/calosum/bootstrap/settings.py` | Feature flag `vector_quantization: str = "none"` |
| `src/calosum/bootstrap/factory.py` | Instanciar e injetar codec quando flag ≠ `"none"` |
| `src/calosum/harness_checks.py` | Registrar `adapters.quantized_embeddings` e `adapters.memory_qdrant_serializers` em `MODULE_RULES` |
| `docs/exec-plans/tech-debt-tracker.md` | Atualizar status / debt novo se surgir |

### Fora de Escopo

- Compressão do KV cache dos Transformers locais (hemisfério esquerdo) — Sprint futura.
- Integração com Qdrant Binary Quantization / HNSW tuning — Sprint futura.
- Fine-tuning / LoRA com pesos quantizados — pós-v1.

---

## Sprints

### Sprint 0 — Qdrant Scalar Quantization (sem código novo)

**Objetivo:** Ganho imediato de 4× no tamanho do índice em disco via configuração.

**Passos:**
1. Em `QdrantAdapterConfig`, adicionar campo `scalar_quantization: bool = False`.
2. Em `_ensure_collections()`, criar coleções com `QuantizationConfig(scalar=ScalarQuantizationConfig(type=ScalarType.INT8))` quando flag ativa.
3. Feature flag em `settings.py`: `CALOSUM_QDRANT_SCALAR_QUANTIZATION=false`.
4. Teste: verificar que a coleção criada com flag ativa tem `quantization_config` não-nula.

**Critério de saída:** `test_qdrant_adapter.py` passa; harness passes.

---

### Sprint 1 — `VectorCodecPort` + `TurboQuantVectorCodec` (núcleo)

**Objetivo:** Implementar os algoritmos PolarQuant e QJL como codec reutilizável.

**`shared/ports.py` — VectorCodecPort:**
```python
@runtime_checkable
class VectorCodecPort(Protocol):
    """Codec de compressão/descompressão de vetores de alta dimensão."""
    def encode(self, vector: list[float]) -> bytes: ...
    def decode(self, compressed: bytes) -> list[float]: ...
    def inner_product_approx(
        self, query: list[float], compressed: bytes
    ) -> float: ...
    @property
    def bits_per_dim(self) -> int: ...
```

**`adapters/quantized_embeddings.py` — TurboQuantVectorCodec:**

Classe principal com dois sub-estágios:

1. **PolarQuantEncoder** (usa maioria dos bits):
   - Converte vetor Cartesiano em coordenadas polares recursivamente (pares de dimensões → raio + ângulo)
   - Quantiza cada ângulo em `bits` níveis uniformes no intervalo `[0, 2π]`
   - Empacota em `bytes` sem overhead de constantes (data-oblivious)

2. **QJLResidualEncoder** (1 bit por dimensão):
   - Aplica random rotation via Hadamard-Walsh rademacher (sem armazenar a matriz — usa semente determinista)
   - Extrai apenas o bit de sinal do vetor rotacionado
   - Estimador unbiased do produto interno

3. **TurboQuantVectorCodec** (composição):
   - `encode(v)` → PolarQuant(v, bits=3) + QJL(residual, bits=1) → bytes compactos
   - `decode(b)` → PolarQuant decode + QJL decode (aprox.)
   - `inner_product_approx(q, b)` → combinação ponderada das duas estimativas
   - `bits_per_dim` = 4 (padrão)

**Limite de linhas:** O módulo deve ficar abaixo de 400 linhas. Se ultrapassar, extrair
`polar_quant.py` e `qjl_encoder.py` como módulos privados.

**Critério de saída:** `test_quantized_embeddings.py` com ≥ 10 casos passa; harness passes.

---

### Sprint 1.5 — Resolução de Debt: Extração de `memory_qdrant_serializers.py`

> **Debt de origem:** `memory_qdrant.py` está em **393/400 linhas** (zona de risco no `tech-debt-tracker.md`).  
> Qualquer linha nova adicionada na Sprint 2 violaria o limite de 400 linhas e quebraria o harness.  
> Esta sprint resolve o debt antes de qualquer integração nesse módulo.

**Objetivo:** Extrair os helpers de serialização de `memory_qdrant.py` para um módulo dedicado, liberando espaço para as mudanças da Sprint 2 sem violar o limite de 400 linhas.

**Passos:**
1. Criar `src/calosum/adapters/memory_qdrant_serializers.py` com as funções:
   - `episode_payload(episode: MemoryEpisode) -> dict`
   - `episode_from_point(point) -> MemoryEpisode`
   - `rule_from_point(point) -> SemanticRule`
   - `rule_document(rule: SemanticRule) -> str`
   - `episode_document(episode: MemoryEpisode) -> str`
   - Funções internas auxiliares: `_parse_datetime`, `_placeholder_right_state`, `_placeholder_bridge_packet`, `_placeholder_left_result`
2. Em `memory_qdrant.py`, substituir os métodos privados equivalentes por imports do novo módulo.
3. Verificar que `memory_qdrant.py` fica abaixo de **370 linhas** (margem para Sprint 2).
4. Registrar `adapters.memory_qdrant_serializers` em `MODULE_RULES` no harness com imports permitidos: `shared.*`.
5. Atualizar `tech-debt-tracker.md`: mover `adapters/memory_qdrant.py` de **Atenção** para **Resolvidos**.

**Critério de saída:**
- `wc -l src/calosum/adapters/memory_qdrant.py` < 370
- `test_qdrant_adapter.py` passa sem alterações (zero breaking changes)
- `PYTHONPATH=src python3 -m calosum.harness_checks` passa

---

### Sprint 2 — Integração em `text_embeddings.py` e `memory_qdrant.py`

**Objetivo:** Wrapear o pipeline de embeddings existente com o codec.

**`text_embeddings.py`:**
- Adicionar parâmetro opcional `codec: VectorCodecPort | None = None` ao `TextEmbeddingAdapter.__init__`
- Novo método `embed_texts_compressed(texts) -> list[bytes]` que chama `aembed_texts` + `codec.encode`
- Backward compatible: sem codec, comportamento idêntico ao atual

**`memory_qdrant.py`:**
- Adicionar `codec: VectorCodecPort | None = None` ao `QdrantDualMemoryAdapter.__init__`
- Em `astore_episode`: se codec presente, armazenar `latent_vector_compressed` no payload (bytes → base64)
- Em `abuild_context`: usar `codec.inner_product_approx` na relevância se disponível
- Usar funções de `memory_qdrant_serializers` (já extraídas na Sprint 1.5)

**Critério de saída:** `test_qdrant_adapter.py` + `test_text_embeddings.py` passam; harness passes.

---

### Sprint 3 — `right_hemisphere_hf.py` + Feature Flag no Bootstrap

**Objetivo:** Acelerar `_calculate_surprise()` e expor controle via settings.

**`right_hemisphere_hf.py`:**
- Injetar `codec: VectorCodecPort | None = None` no `__init__`
- Em `_calculate_surprise()`: se codec disponível, usar `codec.inner_product_approx` ao invés
  de `np.dot` float32 — mantendo fallback para o caminho atual
- Telemetria: adicionar `"codec_used": bool` ao dicionário de telemetria no `RightHemisphereState`

**`bootstrap/settings.py`:**
```python
CALOSUM_VECTOR_QUANTIZATION: str = "none"   # "none" | "turboquant"
CALOSUM_TURBOQUANT_BITS: int = 4
CALOSUM_QDRANT_SCALAR_QUANTIZATION: bool = False
```

**`bootstrap/factory.py`:**
```python
# Em build_right_hemisphere() / build_memory()
codec = None
if settings.vector_quantization == "turboquant":
    from calosum.adapters.quantized_embeddings import TurboQuantVectorCodec
    codec = TurboQuantVectorCodec(bits=settings.turboquant_bits)
```

**Critério de saída:** `test_right_hemisphere_hf.py` + `test_factory.py` passam; harness passes.

---

## Validation

### Testes Unitários (obrigatório por sprint)

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -t .
```

#### Novos cenários em `tests/test_quantized_embeddings.py`

| ID | Descrição | Tipo |
|---|---|---|
| `test_encode_decode_roundtrip` | `decode(encode(v))` ≈ `v` com erro coseno < 5% | Propriedade |
| `test_inner_product_sign_preserved` | `sign(inner_product_approx(q, encode(v)))` == `sign(dot(q, v))` | Propriedade |
| `test_encode_output_is_bytes` | Output de `encode` é `bytes` | Unidade |
| `test_bits_per_dim_property` | `bits_per_dim` retorna inteiro ≥ 1 | Unidade |
| `test_zero_vector_stable` | Vetor zero não gera exceção | Edge case |
| `test_unit_vector_stable` | Vetor unitário comprime e descomprime sem NaN | Edge case |
| `test_random_vectors_recall` | 100 vetores aleatórios: recall@1 ≥ 90% | Benchmark interno |
| `test_codec_port_protocol` | `TurboQuantVectorCodec` satisfaz `isinstance(c, VectorCodecPort)` | Contrato |
| `test_compressed_size_reduction` | `len(encode(v)) < len(v) * 4` (float32 = 4 bytes/dim) | Regressão |
| `test_qdrant_scalar_quantization_flag` | Coleção criada com flag tem `quantization_config` não-nula | Integração |

#### Cenários adicionados em testes existentes

| Arquivo | Cenário adicionado |
|---|---|
| `test_qdrant_adapter.py` | `test_serializers_roundtrip` — `episode_from_point(point_from_episode(ep))` preserva campos críticos (Sprint 1.5) |
| `test_qdrant_adapter.py` | `test_no_regression_after_extraction` — todos os testes existentes passam após extração dos serializers |
| `test_text_embeddings.py` | `test_embed_with_codec` — codec injeta compressão transparente |
| `test_qdrant_adapter.py` | `test_store_with_codec_payload` — payload contém `latent_vector_compressed` |
| `test_right_hemisphere_hf.py` | `test_surprise_with_codec` — codec acelera surprise sem mudar sinal |
| `test_factory.py` | `test_factory_turboquant_flag` — env `CALOSUM_VECTOR_QUANTIZATION=turboquant` instancia codec |

### Harness Check (obrigatório ao final de cada sprint)

```bash
PYTHONPATH=src python3 -m calosum.harness_checks
```

**Checagens relevantes para este plano:**

| Check | O que valida |
|---|---|
| `missing_module_rule` | `adapters.quantized_embeddings` e `adapters.memory_qdrant_serializers` registrados em `MODULE_RULES` |
| `forbidden_internal_import` | `quantized_embeddings` e `memory_qdrant_serializers` só importam de `shared.*` |
| `module_too_large` | Nenhum módulo modificado ultrapassou 400 linhas |
| `plan_missing_heading` | Este plano contém todos os headings obrigatórios |

**Sequência de validação final (após Sprint 3):**

```bash
# 1. Testes completos
PYTHONPATH=src python3 -m unittest discover -s tests -t .

# 2. Harness
PYTHONPATH=src python3 -m calosum.harness_checks

# 3. Verificação de tamanho dos módulos modificados
wc -l src/calosum/adapters/quantized_embeddings.py
wc -l src/calosum/adapters/memory_qdrant.py
wc -l src/calosum/adapters/memory_qdrant_serializers.py
wc -l src/calosum/shared/ports.py
wc -l src/calosum/bootstrap/settings.py
wc -l src/calosum/bootstrap/factory.py
```

### Benchmarks de Recall (opcional mas recomendado)

Extender `scripts/benchmark_right_hemisphere.py` com seção TurboQuant:

```bash
PYTHONPATH=src .venv/bin/python3 scripts/benchmark_right_hemisphere.py --turboquant
```

Métrica alvo: **recall@1 ≥ 90%** em vetores aleatórios 384-dim, compressão **≥ 6×** vs fp32.

---

## Progress

- [ ] **Sprint 0** — Qdrant Scalar Quantization via config
- [ ] **Sprint 1** — `VectorCodecPort` em `shared/ports.py` + `TurboQuantVectorCodec` em `adapters/quantized_embeddings.py`
- [ ] **Sprint 1.5** — Debt: extrair serializers de `memory_qdrant.py` → `memory_qdrant_serializers.py`; registrar no harness; fechar item no `tech-debt-tracker.md`
- [ ] **Sprint 2** — Integração em `text_embeddings.py` e `memory_qdrant.py`
- [ ] **Sprint 3** — Integração em `right_hemisphere_hf.py` + Feature flags em `settings.py` + `factory.py`
- [ ] **Validação final** — `unittest` + `harness_checks` + `wc -l` nos módulos modificados
- [ ] **Fechamento** — Mover para `completed/`, atualizar `tech-debt-tracker.md`

---

## Decision Log

| Data | Decisão | Motivo |
|---|---|---|
| 2026-03-31 | Implementar PolarQuant/QJL do zero em numpy | Código oficial ainda não liberado; algoritmos são matematicamente acessíveis; zero dependência nova |
| 2026-03-31 | Sprint 0 separada (Qdrant SQ) | Ganho imediato sem risco; desacopla entrega incremental |
| 2026-03-31 | Debt de `memory_qdrant.py` incorporado como Sprint 1.5 | O `tech-debt-tracker.md` exige extração antes de qualquer feature; tornamos o debt parte do plano para garantir rastreabilidade e critério de saída formal |
| 2026-03-31 | KV cache de Transformers fora do escopo | Risco médio, dependência de versão HF; mover para sprint separada quando TurboQuant for publicado oficialmente |
| 2026-03-31 | Feature flag `CALOSUM_VECTOR_QUANTIZATION` | Permite rollout incremental e reversão sem mudança de código |
