# Calosum

Esqueleto em Python para um agente neuro-simbolico com:

- hemisferio direito orientado a percepcao e estado latente;
- corpo caloso que comprime intuicao em soft prompts;
- hemisferio esquerdo orientado a raciocinio tipado e acoes primitivas;
- memoria dual com consolidacao episodica -> semantica em sleep mode.
- runtime estrito para acoes primitivas tipificadas;
- telemetria cognitiva separada por hemisferio e decisao;
- reflexao metacognitiva em grupo no estilo GEA.
- CLI para execucao de turnos ou cenarios completos em JSON.
- testes unitarios e integrados da pipeline cognitiva.

## Estrutura

- `src/calosum/types.py`: tipos de comunicacao entre os modulos
- `src/calosum/right_hemisphere.py`: interface do modelo de mundo JEPA
- `src/calosum/bridge.py`: tokenizacao cognitiva e interruptor de saliencia
- `src/calosum/left_hemisphere.py`: SLM logico com saida em acoes tipificadas
- `src/calosum/memory.py`: memoria episodica, memoria semantica e sleep mode
- `src/calosum/runtime.py`: validacao e execucao segura das acoes tipificadas
- `src/calosum/telemetry.py`: barramento e paines de telemetria cognitiva
- `src/calosum/serialization.py`: serializacao JSON-safe dos artefatos cognitivos
- `src/calosum/cli.py`: interface de linha de comando para turnos e cenarios
- `src/calosum/metacognition.py`: avaliacao de variantes cognitivas e auto-ajuste
- `src/calosum/orchestrator.py`: ciclo cognitivo de ponta a ponta
- `examples/cognitive_cycle.py`: exemplo executavel
- `examples/group_reflection.py`: exemplo com multiplas variantes cognitivas
- `deploy/docker-compose.yml`: topologia de contêineres proposta
- `tests/`: suite de testes automatizados

## Execucao do exemplo

```bash
PYTHONPATH=src python3 examples/cognitive_cycle.py
PYTHONPATH=src python3 examples/group_reflection.py
PYTHONPATH=src python3 -m calosum.cli run-turn --session-id demo --text "Estou ansioso e preciso de um plano urgente"
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Formato de cenario JSON

```json
{
  "session_id": "demo-session",
  "turns": [
    {
      "text": "Estou frustrado e preciso de um plano urgente.",
      "signals": [
        {
          "modality": "audio",
          "source": "microphone",
          "payload": {"transcript": "voz tensa"},
          "metadata": {"emotion": "frustrado"}
        }
      ]
    },
    {
      "text": "Prefiro respostas curtas com passos claros.",
      "group_variants": [
        {"variant_id": "empathetic", "tokenizer_overrides": {"salience_threshold": 0.45}},
        {"variant_id": "strict", "tokenizer_overrides": {"salience_threshold": 0.9}}
      ]
    }
  ],
  "sleep_mode": true
}
```
