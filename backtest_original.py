import os
import pandas as pd
import numpy as np

# Passo 1 - abrir as bases
precos = pd.read_parquet("precos_limpo.parquet")
roe    = pd.read_parquet("roe_limpo.parquet")
mktcap = pd.read_parquet("mktcap_limpo.parquet")
ineg   = pd.read_parquet("ineg_limpo.parquet")

# ibov e setores vem do limpando_dados.py
tem_ibov = os.path.exists("ibov.parquet")
if tem_ibov:
    ibov = pd.read_parquet("ibov.parquet").iloc[:, 0]

print(precos.shape, roe.shape, mktcap.shape, ineg.shape)

# Passo 2 - os parametros
data_inicial = pd.Timestamp("2010-01-04")
data_final   = min(precos.index[-1], mktcap.index[-1], ineg.index[-1])

n_liquidas   = 100   # quantas empresas entram no universo
meses_rebal  = 1     # de quanto em quanto tempo troca o bloco agressivo
meses_cons   = 3     # de quanto em quanto tempo troca o bloco conservador
meses_liq    = 3     # janela para medir liquidez e achar o market cap
meses_vol    = 12    # janela da volatilidade
meses_mom    = 12    # janela do momentum
meses_pula   = 1     # ultimo mes fora do momentum
dias_roe     = 90    # espera ate o balanco ser publico

peso_cons    = 0.6   # bloco conservador (Quality + Low Vol)
peso_agr     = 0.4   # bloco agressivo (Size + Momentum)
n_cons       = 15    # acoes no bloco conservador
n_size       = 30    # menores do universo
n_agr        = 10    # acoes no bloco agressivo
custo        = 0.001 # 0,1% no primeiro dia de cada rebalanceamento

# Passo 4 - o loop
retornos = pd.DataFrame()
carteiras = {}
data = data_inicial
volta = 0

while data + pd.DateOffset(months=meses_rebal) <= data_final:

    proxima = data + pd.DateOffset(months=meses_rebal)
    trimestre = volta % (meses_cons // meses_rebal) == 0

    # 1. universo
    liquidez = ineg.loc[data - pd.DateOffset(months=meses_liq):data].ffill()
    cap = mktcap.loc[data - pd.DateOffset(months=meses_liq):data].ffill()
    if liquidez.empty or cap.empty:
        data = proxima
        continue
    cap = cap.iloc[-1].dropna()

    liquidez = liquidez.iloc[-1].dropna()
    classes = liquidez.groupby(liquidez.index.str[:4]).idxmax().values
    elegiveis = liquidez[classes].sort_values(ascending=False).head(n_liquidas).index

    colunas = precos.columns.intersection(elegiveis)

    # 2. bloco conservador: top 15 da media dos z, troca a cada trimestre
    if trimestre:
        janela_vol = precos.loc[data - pd.DateOffset(months=meses_vol):data, colunas]
        vol = janela_vol.pct_change().std()

        corte = data - pd.Timedelta(days=dias_roe)
        roe_data = roe.loc[corte - pd.DateOffset(months=12):corte].ffill().iloc[-1]

        fatores = pd.DataFrame({"roe": roe_data.reindex(colunas), "vol": vol}).dropna()
        if len(fatores) < n_cons:
            data = proxima
            continue
        z_roe = (fatores["roe"] - fatores["roe"].mean()) / fatores["roe"].std()
        z_vol = -(fatores["vol"] - fatores["vol"].mean()) / fatores["vol"].std()
        nota = ((z_roe + z_vol) / 2).sort_values(ascending=False)
        conservador = nota.index[:n_cons]

    # 3. bloco agressivo: 30 menores, depois top 10 de momentum 12-1
    menores = cap.reindex(colunas).dropna().sort_values().index[:n_size]
    janela_mom = precos.loc[data - pd.DateOffset(months=meses_mom):
                            data - pd.DateOffset(months=meses_pula), menores]
    if len(janela_mom) < 2:
        data = proxima
        continue
    momentum = (janela_mom.iloc[-1] / janela_mom.iloc[0] - 1).dropna()
    momentum = momentum.sort_values(ascending=False)
    if len(momentum) < n_agr:
        data = proxima
        continue
    agressivo = momentum.index[:n_agr]

    # 4. retorno do mes seguinte
    janela = precos.loc[data:proxima]
    def retorno_carteira(cols):
        return janela[cols].pct_change().mean(axis=1)[1:]

    serie_cons = retorno_carteira(conservador)
    serie_agr  = retorno_carteira(agressivo)
    cons_custo = serie_cons.copy()
    agr_custo  = serie_agr.copy()
    if trimestre:
        cons_custo.iloc[0] -= custo
    agr_custo.iloc[0] -= custo

    retornos = pd.concat([retornos, pd.DataFrame({
        "Estrategia":  peso_cons * serie_cons + peso_agr * serie_agr,
        "Com custo":   peso_cons * cons_custo + peso_agr * agr_custo,
        "Conservador": serie_cons,
        "Agressivo":   serie_agr,
        "Universo":    retorno_carteira(colunas),
    })])

    # pesos somam quando a empresa esta nos dois blocos
    peso = pd.Series(peso_cons / n_cons, index=conservador)
    carteiras[data] = peso.add(pd.Series(peso_agr / n_agr, index=agressivo), fill_value=0)

    # 5. anda um mes
    data = proxima
    volta += 1

if tem_ibov:
    retornos["Ibovespa"] = ibov.reindex(precos.index).pct_change().reindex(retornos.index)

print(retornos.shape)
print(retornos.index[0].date(), "a", retornos.index[-1].date())

# Passo 5 - medir o resultado
curva = retornos.add(1).cumprod()

print(curva.iloc[-1].round(2))

anos = (curva.index[-1] - curva.index[0]).days / 365.25

for c in curva.columns:
    cagr  = curva[c].iloc[-1] ** (1 / anos) - 1
    vol   = retornos[c].std() * np.sqrt(252)
    queda = (curva[c] / curva[c].cummax() - 1).min()
    print(f"{c:12s} CAGR={cagr:6.2%} vol={vol:5.1%} Sharpe={cagr/vol:5.2f} queda={queda:6.1%}")

# o que a apresentacao pediu para acompanhar
pesos = pd.DataFrame(carteiras).T.fillna(0)
giro  = pesos.diff().abs().sum(axis=1).iloc[1:]
parte = peso_agr * retornos["Agressivo"]
risco = parte.cov(retornos["Estrategia"]) / retornos["Estrategia"].var()

print("rebalanceamentos:", len(pesos))
print("posicoes: ", int((pesos > 0).sum(axis=1).min()), "a", int((pesos > 0).sum(axis=1).max()))
print("giro/ano: ", f"{giro.sum() / anos:.0%}")
print("risco do agressivo:", f"{risco:.0%}")

# concentracao setorial
if os.path.exists("setores.parquet"):
    setores = pd.read_parquet("setores.parquet")["Setor"]
    setores = setores.reindex(pesos.columns).fillna("Sem setor")
    por_setor = pesos.T.groupby(setores).sum().T

    print(por_setor.mean().sort_values(ascending=False).round(3))
    print("maior setor na media das datas:", f"{por_setor.max(axis=1).mean():.0%}")
    print("maior setor em uma data:       ", f"{por_setor.max(axis=1).max():.0%}")

# testando os pesos entre os blocos, sem custo
for p in [0.4, 0.5, 0.6, 0.7, 0.8]:
    serie = p * retornos["Conservador"] + (1 - p) * retornos["Agressivo"]
    curva_p = serie.add(1).cumprod()
    cagr  = curva_p.iloc[-1] ** (1 / anos) - 1
    vol   = serie.std() * np.sqrt(252)
    queda = (curva_p / curva_p.cummax() - 1).min()
    print(f"{p:.0%}/{1 - p:.0%}  CAGR={cagr:6.2%} vol={vol:5.1%} Sharpe={cagr/vol:5.2f} queda={queda:6.1%}")

retornos.to_parquet("retornos_backtest.parquet")
pesos.to_parquet("carteiras_backtest.parquet")
