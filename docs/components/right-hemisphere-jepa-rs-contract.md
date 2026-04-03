# JEPA-RS Contract

## Purpose

Formalizar o contrato operacional do backend Rust `jepa-rs` consumido pelo adapter Python `input_perception_jepars.py`.

## Contract Version

- `jepa-rs-arrow-v1`
- transporte: `arrow_stream`
- invocacao atual: `jepa-rs infer --arrow`

## Input Payload

O adapter envia JSON em `stdin` com:

- `text`: texto bruto do turno
- `signals_count`: quantidade de sinais multimodais recebidos
- `model_path`: caminho opcional de checkpoint local
- `latent_size`: dimensionalidade esperada do vetor
- `contract_version`: `jepa-rs-arrow-v1`
- `transport`: `arrow_stream`

## Output Payload

Campos obrigatorios:

- `latent_vector`: `list[number]`, nao vazia

Campos opcionais:

- `salience`: `number`
- `confidence`: `number`
- `surprise_score`: `number`
- `emotional_labels`: `list[str]`
- `latent_mu`: `list[number]`
- `latent_logvar`: `list[number]`

## Adapter Guarantees

- valida `latent_vector` como lista numerica antes de materializar `InputPerceptionState`
- normaliza a telemetria para `right_backend=jepars_local`
- expoe `contract_version`, `checkpoint_loaded` e `multimodal_active`
- falha com erro explicito se o backend retornar schema invalido ou stream Arrow vazio

## Readiness And Telemetry

- `/ready` deve refletir o backend ativo e o budget operacional aplicado
- telemetria por turno deve expor `contract_version`
- qualquer fallback por budget deve aparecer como `degraded_reason=budget_exceeded:<backend>`
