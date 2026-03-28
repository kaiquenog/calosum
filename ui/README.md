# Calosum UI

Painel React para visualizacao da telemetria cognitiva exposta pela API do Calosum.

## Comandos

- `npm run dev`: sobe o painel em modo de desenvolvimento
- `npm run build`: valida tipos e gera o bundle de producao
- `npm run lint`: executa o lint TypeScript/React

## Integracao

- por padrao, o frontend consome `http://localhost:8000`
- para alterar o backend, defina `VITE_CALOSUM_API_BASE`
- por padrao, o painel observa a sessao `terminal-session`
- o session id escolhido fica salvo no navegador
- o dashboard faz polling automatico da API para refletir novos turnos sem refresh manual

## Escopo Atual

- exibe eventos `felt`, `thought`, `decision`, `execution` e `reflection`
- serve como painel de inspeção operacional, nao como chat cliente final
