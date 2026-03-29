# Plano de Execução: Evolução do Hemisfério Direito e Aprendizado Contínuo (Caminho A)

## Purpose
O objetivo deste plano é amadurecer o **Hemisfério Direito** (System 1) para que ele deixe de ser um processador superficial de emoções e passe a influenciar dinamicamente o raciocínio do agente, além de preparar o terreno para o aprendizado contínuo (Sleep Mode).

## Scope
1. **Percepção Realista**: Substituir *stubs* matemáticos por inferências semânticas reais e aprimorar a extração de emoções (Zero-Shot).
2. **Modulação Cognitiva**: Conectar o nível de "Surpresa" (Free Energy) gerado pelo Hemisfério Direito à temperatura do LLM no Orquestrador.
3. **Fundação do Aprendizado (Sleep Mode)**: Criar a base para extração de memórias de sucesso visando futuro *fine-tuning* (LoRA) ou otimização de *prompts* (DSPy).

## Validation
- `PYTHONPATH=src python3 -m calosum.harness_checks` passa sem erros de violação de arquitetura.
- Os testes unitários passam (`PYTHONPATH=src python3 -m unittest discover -s tests -t .`).
- O log cognitivo demonstra a variação da temperatura do LLM conforme o nível de surpresa do input do usuário.
- O script/módulo do Sleep Mode consegue gerar um arquivo `.jsonl` válido a partir das memórias existentes.

## Progress
- [x] Passo 1: Remoção de Stubs e Evolução da Inferência Emocional
- [x] Passo 2: Modulação Dinâmica de Temperatura via "Surpresa"
- [x] Passo 3: Fundação do Sleep Mode (Consolidação Noturna)

## Decision Log
- Decidimos usar o desvio padrão e heurística em vez de um modelo de zero-shot completo para evitar overengineering e manter a performance local sem GPU.
- Modulamos a temperatura de forma linearmente proporcional ao `surprise_score`.