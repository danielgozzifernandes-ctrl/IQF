# IQF - backtest de estratégia de fatores (Brasil)

Backtest de uma carteira 60% **Conservador** (Quality + Low Vol) e 40% **Agressivo**
(Size + Momentum), nas 100 ações mais líquidas da B3, de jan/2010 a out/2026.

**Regra do projeto:** a estratégia não é ajustada para dar certo. Ela pode dar certo
ou errado; o trabalho é explicar **por que** deu o resultado que deu. Corrigir erros
de medição e acrescentar análises pode; mudar parâmetros para melhorar o número, não.
Detalhes em [CLAUDE.md](CLAUDE.md).

## Como rodar

```bash
pip install -r requirements.txt
python limpando_dados.py   # só se os .xlsx mudarem: gera os *_limpo.parquet
python backtest.py         # roda o backtest, imprime as medidas e gera os gráficos
```

Na primeira execução o `backtest.py` baixa o CDI diário do Banco Central
(`cdi.parquet`, série 12 do SGS). Para baixar de novo, apague o arquivo.
A execução leva cerca de 1 minuto, porque roda a grade de robustez (8 versões).

## O que sai

| Arquivo | O que é |
|---|---|
| `relatorio_quantstats.html` | Relatório completo do QuantStats (abra no navegador) contra o Ibovespa, `rf` = CDI médio |
| `graficos/*.png` | Curva acumulada, drawdown, retorno anual, retorno acima do CDI, teste de pesos, setores, grade de robustez |
| `resultados/*.csv` | Tabelas usadas no painel: medidas, curva, retorno anual, decomposição, fases, setores, sorte, custos, robustez |
| `retornos_backtest.parquet` | Retornos diários de cada série |
| `carteiras_backtest.parquet` | Pesos de cada ação em cada rebalanceamento |
| `explica_backtest.parquet` | Diferença para o Ibovespa quebrada em peso igual, Conservador, Agressivo e custo |

## Arquivos de entrada

- `fechamento.xlsx`, `ROE.xlsx`, `marketcap.xlsx`, `ineg.xlsx`: exportações do Economatica.
- `ibov_yahoo.xlsx`, `setores_yahoo.xlsx`: Yahoo Finance.
- `base_antiga/`: versão anterior das bases, guardada para comparação.
- `backtest_original.py`: versão do backtest antes da correção dos feriados.

## Como ler as medidas

- **Sharpe**: retorno acima do CDI dividido pela volatilidade. Responde "valeu correr risco em vez de ficar no CDI?".
- **Beta / alfa / IR**: comparação com o Ibovespa. Beta = quanto anda junto; alfa = o que sobra além do beta; IR = ganho sobre o Ibovespa dividido pela distância dele.

## As perguntas da Aula 3 (Análise de Resultados)

| Pergunta | Onde está |
|---|---|
| Ganhou do CDI? | Medidas impressas, `resultados/medidas.csv`, QuantStats |
| Tem anos suficientes de dado? (foi sorte?) | Estatística t = Sharpe × √anos e anos para t = 2: `resultados/sorte.csv` |
| Mudando os parâmetros, o resultado se mantém? | Grade momentum (3, 6, 9, 12 meses) × rebal (mensal, trimestral): `resultados/robustez.csv`, `graficos/7_robustez.png` |
| Descontei os custos? | Custo oficial (0,1% por rebal) e regra da aula (0,2% × giro): `resultados/custos.csv` |

Esses testes não mudam a estratégia oficial (momentum 12m, rebal mensal, 60/40); servem para explicar o resultado.

## Painel de resultados

https://claude.ai/artifact/XwJRoutBYw7JDfinwLSKDC (privado do dono; os dados dele são os CSVs de `resultados/`).

## Armadilhas conhecidas

- O Economatica traz feriados como linhas inteiras vazias. O backtest descarta essas linhas; sem isso, 15 meses sumiam do resultado (inclusive mar/2020).
- `setores.parquet` deixa ~25% do peso como "Sem setor".
- Os dados do Economatica são licenciados: usem só para o trabalho do grupo.
