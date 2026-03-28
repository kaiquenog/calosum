# Calosum UI

Painel React para visualizacao da telemetria cognitiva exposta pela API do Calosum.

## Comandos

- `npm run dev`: sobe o painel em modo de desenvolvimento
- `npm run build`: valida tipos e gera o bundle de producao
- `npm run lint`: executa o lint TypeScript/React

## Integracao

- por padrao, o frontend consome `http://localhost:8000`
- para alterar o backend, defina `VITE_CALOSUM_API_BASE`

## Escopo Atual

- exibe eventos `felt`, `thought`, `decision`, `execution` e `reflection`
- serve como painel de inspeção operacional, nao como chat cliente final
