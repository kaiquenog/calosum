# Plano de Execução: Fase 4 - Auto-Aprendizado (DSPy e LoRA)

## Purpose
Implementar a infraestrutura de aprendizado contínuo (Sleep Mode) do Calosum, fechando o ciclo neuro-simbólico. O objetivo é que o agente use as experiências consolidadas durante o dia para otimizar seus próprios prompts (DSPy) e refinar os pesos do seu modelo base (LoRA).

## Scope
1. **Otimização de Prompts (DSPy)**: 
   - Criar ou refinar o script `night_trainer_dspy.py` para consumir o arquivo `dspy_dataset.jsonl`.
   - Configurar o otimizador `BootstrapFewShot` ou `MIPROv2` para extrair os melhores *few-shots* das interações diárias.
   - Atualizar a configuração dinâmica do `Orchestrator` para carregar o prompt otimizado na manhã seguinte.

2. **Treinamento Contínuo de Pesos (LoRA)**:
   - Criar o script autônomo `lora_trainer.py` que consome o dataset `lora_sharegpt.jsonl`.
   - Utilizar a biblioteca `peft` e `transformers` (ou um wrapper leve como o `unsloth` se possível) para realizar o *fine-tuning* do modelo LLM local (ex: Qwen) de forma quantizada (4-bit/8-bit).
   - Salvar o novo adaptador na pasta de artefatos.
   - Preparar o `LeftHemisphereLogicalSLM` para carregar os pesos LoRA atualizados no momento do boot.

## Validation
- O script DSPy deve ler o `.jsonl` e gerar um arquivo JSON com os few-shots escolhidos.
- O script LoRA deve iniciar o treinamento local sem estourar a memória e gerar um arquivo `.safetensors` ou `.bin` na pasta de adaptadores.
- O projeto continuará passando nas verificações de `harness_checks.py`.
- Os testes unitários existentes e os novos referentes aos trainers devem passar com sucesso.

## Progress
- [x] Implementar Otimização DSPy.
- [x] Implementar Treinamento LoRA.
- [x] Injetar carregamento no Hemisfério Esquerdo.

## Decision Log
- Decidimos separar a execução noturna em scripts autônomos para não sobrecarregar a API principal durante o dia.
- O treinamento LoRA deve suportar fallback (pular o passo) caso não haja GPU disponível no ambiente local.