# flower-test: Exemplo Flower + PyTorch para Aprendizado Federado

Este repositório contém um exemplo didático que usa Flower (FLWR) e PyTorch para demonstrar conceitos e experimentos de aprendizado federado. O objetivo é fornecer um ponto de partida para experimentar estratégias de agregação (por exemplo, FedAvg, FedProx), verificar o efeito de parâmetros (como o termo de proximidade mu em FedProx) e salvar/avaliar modelos e métricas de treino.

## O que é Aprendizado Federado (visão geral curta)

O Aprendizado Federado (Federated Learning, FL) é um paradigma de treinamento distribuído onde múltiplos dispositivos (clientes) treinam localmente um modelo usando seus dados locais e apenas enviam atualizações (por exemplo, pesos ou gradientes) a um servidor central para agregação. Isso permite treinar modelos colaborativamente sem centralizar os dados, preservando privacidade e reduzindo a necessidade de transferência de grandes volumes de dados sensíveis.

Principais vantagens:
- Privacidade: dados permanecem nos dispositivos.
- Comunicação eficiente: troca apenas atualizações do modelo.
- Robusto a heterogeneidade: clientes com dados não-iid e diferenças de capacidade.

Desafios comuns:
- Dados não-iid entre clientes.
- Comunicação limitada/ruim.
- Clientes que saem/entram e falham.
- Desequilíbrio no tamanho de dados entre clientes.

## Objetivo deste projeto

- Demonstrar um pipeline simples de FL usando Flower e PyTorch.
- Comparar estratégias/algoritmos (p.ex. FedAvg vs FedProx com diferentes valores de mu).
- Salvar modelos treinados e métricas de cada experimento para análise posterior.
- Fornecer scripts para rodar simulações locais e reproduzir resultados.

## Estrutura do repositório

- `pyproject.toml` — dependências e configuração/hyperparâmetros usados pelo exemplo.
- `README.md` — este arquivo (documentação e instruções).
- Arquivos de resultados e modelos:
  - `train_results_fedavg.json` — métricas geradas durante o experimento FedAvg.
  - `train_results_fedprox_mu=0.1.json`, `train_results_fedprox_mu=1.json`, `train_results_fedprox_my=*.json` — métricas para execuções de FedProx com diferentes valores de mu.
  - `fedavg_model.pt`, `fedprox_model.pt`, `fedprox_model_mu=1.pt`, `fedprox_mu=0.1_model.pt` — modelos PyTorch salvos ao fim de cada experimento.
- Diretório `flower_test/` — implementação do experimento:
  - `task.py` — definição de tarefa: modelo PyTorch, dataset, funções de treinamento/avaliação.
  - `client_app.py` — implementação do cliente Flower que treina localmente e comunica atualizações ao servidor.
  - `server_app.py` — define/lança a simulação/servidor Flower e a estratégia (FedAvg, FedProx, etc).
  - `__init__.py` — marca o pacote Python.

> Observação: os nomes dos arquivos `train_results_*.json` registram métricas por rodada (por exemplo, perda e acurácia por round). Os arquivos `.pt` são checkpoints dos modelos finais.

## Como funciona (fluxo de alto nível)

1. `server_app.py` configura a estratégia de agregação e inicia o orquestrador Flower (simulação local por padrão).
2. `client_app.py` representa o comportamento de um dispositivo cliente:
   - Carrega seus dados locais (definidos em `task.py`).
   - Treina localmente por algumas épocas.
   - Retorna os pesos/atualizações ao servidor.
3. O servidor agrega as atualizações (por exemplo, média ponderada — FedAvg — ou FedProx que adiciona termo de penalidade) e atualiza o modelo global.
4. O processo repete por um número configurado de rounds. Métricas e modelos são salvos a cada execução.

## Como executar (simulação local)

1. Instale o pacote e dependências (recomendado criar um ambiente virtual):
```powershell
pip install -e .
```

2. Rode a simulação local usando o runtime de simulação do Flower:
```powershell
flwr run .
```

- Observação: algumas execuções podem estar configuradas via `pyproject.toml`. Verifique esse arquivo para parâmetros de execução (número de rounds, número de clientes simulados, hiperparâmetros de treinamento, learning rate, etc).
- Alternativamente, use o comando completo para alterar os hiperparâmetros, substitua {value} pelo valor desejado.
```powershell
flwr run . --run-config "num-server-rounds={value} fraction-train={value} local-epochs={value} lr={value} test-name={value} proximal-mu={value}"
```

## Experimentos incluídos

Este repositório já contém resultados e modelos de várias execuções:
- FedAvg: `train_results_fedavg.json`, `fedavg_model.pt`
- FedProx com várias configurações de mu (por ex. mu = 0.1, 1): `train_results_fedprox_*.json`, `fedprox_model*.pt`

Cada arquivo `train_results_*.json` normalmente contém métricas por round, tais como:
- loss_train / loss_eval
- accuracy_train / accuracy_eval
- (possivelmente) tempo por round e outras estatísticas

Use esses arquivos para traçar curvas de perda/acurácia por round e comparar comportamento entre algoritmos.

## Arquivos principais

- `flower_test/task.py` — define:
  - O modelo PyTorch (arquitetura usada).
  - A função de perda, otimizador.
  - Dataset e transformação (simulação de dados dos clientes).
  - Funções auxiliares de treino e avaliação.
- `flower_test/client_app.py` — implementa a interface do cliente Flower:
  - Recebe o modelo global, faz treino local, retorna atualizações.
- `flower_test/server_app.py` — orquestra a simulação:
  - Configura a estratégia (FedAvg / FedProx).
  - Define número de rounds, clientes por rodada, e pontos de checkpoint.
  - Inicia o loop de federated rounds.

## Contrato rápido

- Inputs:
  - Configurações (via `pyproject.toml` ou variáveis em `server_app.py`): n_rounds, num_clients, epochs_local, batch_size, learning_rate, mu (para FedProx), etc.
  - Dados locais simulados (definidos em `task.py`).
- Outputs:
  - Arquivos JSON com métricas (`train_results_*.json`).
  - Modelos salvos (`*.pt`).
- Critério de sucesso:
  - Execução completa sem erro.
  - Geração dos arquivos de métricas e checkpoints.
  - Melhorias observáveis nas métricas do modelo global ao longo dos rounds.

## Casos de borda / limitações importantes

- Dados não-iid podem causar convergência mais lenta e maior variância entre clientes.
- Se clientes tiverem volumes de dados muito desbalanceados, o peso na agregação afeta fortemente o modelo global.
- Experimentos locais (simulação) não reproduzem completamente latências e falhas do mundo real.
- FedProx pode ajudar a estabilizar quando clientes divergem muito, mas precisa de ajuste de mu.

## Como interpretar os resultados

- Plote acurácia e loss por round a partir dos `train_results_*.json`.
- Compare diferentes arquivos para ver o efeito da estratégia ou do parâmetro mu.
- Verifique o tamanho e a qualidade do dataset simulado em `task.py` para entender diferenças entre execuções.

## Referências e leitura adicional

- Flower framework: https://flower.ai
- Papers clássicos:
  - McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
  - Li et al., "Federated Optimization in Heterogeneous Networks" (FedProx)
- Documentação PyTorch: https://pytorch.org*