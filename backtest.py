import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import quantstats as qs

# Passo 1 - abrir as bases
# feriados vem do Economatica como linhas inteiras vazias: sem tirar, o momentum
# comecava num feriado, ficava vazio e o mes inteiro sumia do backtest
precos = pd.read_parquet("precos_limpo.parquet").dropna(how="all")
roe    = pd.read_parquet("roe_limpo.parquet")
mktcap = pd.read_parquet("mktcap_limpo.parquet")
ineg   = pd.read_parquet("ineg_limpo.parquet")

# ibov e setores vem do limpando_dados.py
tem_ibov = os.path.exists("ibov.parquet")
if tem_ibov:
    ibov = pd.read_parquet("ibov.parquet").iloc[:, 0]

# selic diaria do Banco Central (serie 11 do SGS), baixada uma vez e guardada
if not os.path.exists("selic.parquet"):
    partes = []
    for ano in range(2000, pd.Timestamp.today().year + 1, 10):
        url = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json"
               f"&dataInicial=01/01/{ano}&dataFinal=31/12/{ano + 9}")
        partes.append(pd.read_json(url))
    selic = pd.concat(partes)
    selic["data"] = pd.to_datetime(selic["data"], dayfirst=True)
    selic = selic.set_index("data")[["valor"]].rename(columns={"valor": "Selic"}) / 100
    selic.to_parquet("selic.parquet")
selic = pd.read_parquet("selic.parquet")["Selic"]

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
        vol = janela_vol.ffill().pct_change(fill_method=None).std()

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
    # comeca no ultimo pregao ate a data, para nao perder o dia quando ela cai num fim de semana
    inicio = precos.index[precos.index <= data][-1]
    janela = precos.loc[inicio:proxima]
    def retorno_carteira(cols):
        return janela[cols].ffill().pct_change(fill_method=None).mean(axis=1).iloc[1:]

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
    retornos["Ibovespa"] = ibov.reindex(precos.index).ffill().pct_change(fill_method=None).reindex(retornos.index)
# selic do dia (taxa diaria): e o que o caixa rende, a regua do Sharpe
retornos["Selic"] = selic.reindex(retornos.index).ffill()

print(retornos.shape)
print(retornos.index[0].date(), "a", retornos.index[-1].date())

# Passo 5 - medir o resultado
curva = retornos.add(1).cumprod()
anos = (curva.index[-1] - curva.index[0]).days / 365.25

def medidas(serie):
    curva_s = serie.add(1).cumprod()
    cagr  = curva_s.iloc[-1] ** (1 / anos) - 1
    vol   = serie.std() * np.sqrt(252)
    queda = (curva_s / curva_s.cummax() - 1).min()
    # Sharpe de verdade: o que sobra acima da Selic, por unidade de risco
    excesso = serie - retornos["Selic"]
    sharpe = excesso.mean() / excesso.std() * np.sqrt(252) if excesso.std() > 0 else np.nan
    return cagr, vol, sharpe, queda

def contra_ibov(serie):
    # o Sharpe compara com o caixa (Selic); contra o Ibov a pergunta e outra:
    # beta = quanto anda junto, alfa = o que sobra alem do beta,
    # IR (information ratio) = retorno acima do Ibov por unidade de distancia dele
    x, b = serie - retornos["Selic"], retornos["Ibovespa"] - retornos["Selic"]
    beta  = x.cov(b) / b.var()
    alfa  = (x - beta * b).mean() * 252
    dif   = serie - retornos["Ibovespa"]
    ir    = dif.mean() / dif.std() * np.sqrt(252) if dif.std() > 0 else np.nan
    return beta, alfa, ir

print(curva.iloc[-1].round(2))
for c in curva.columns:
    cagr, vol, sharpe, queda = medidas(retornos[c])
    linha = f"{c:12s} CAGR={cagr:6.2%} vol={vol:5.1%} Sharpe={sharpe:5.2f} queda={queda:6.1%}"
    if tem_ibov and c != "Selic":
        beta, alfa, ir = contra_ibov(retornos[c])
        linha += f" | beta={beta:4.2f} alfa={alfa:6.2%} IR={ir:5.2f}"
    print(linha)

anual = retornos.add(1).groupby(retornos.index.year).prod() - 1
print((anual * 100).round(1))
print("anos acima da Selic:", int((anual["Com custo"] > anual["Selic"]).sum()), "de", len(anual))
if tem_ibov:
    print("anos acima do Ibov: ", int((anual["Com custo"] > anual["Ibovespa"]).sum()), "de", len(anual))

# de onde veio a diferenca para o Ibov, ano a ano (aproximado, soma simples)
# Ibov -> Universo: so por pegar as 100 mais liquidas com peso igual
# Universo -> blocos: o que cada fator acrescentou, na proporcao do seu peso
if tem_ibov:
    explica = pd.DataFrame({
        "Peso igual":  anual["Universo"] - anual["Ibovespa"],
        "Conservador": peso_cons * (anual["Conservador"] - anual["Universo"]),
        "Agressivo":   peso_agr * (anual["Agressivo"] - anual["Universo"]),
        "Custo":       anual["Com custo"] - anual["Estrategia"],
    })
    explica["Total"] = anual["Com custo"] - anual["Ibovespa"]
    print((explica * 100).round(1))
    print("media por ano:", (explica.mean() * 100).round(1).to_dict())
    explica.to_parquet("explica_backtest.parquet")

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
tem_setores = os.path.exists("setores.parquet")
if tem_setores:
    setores = pd.read_parquet("setores.parquet")["Setor"]
    setores = setores.reindex(pesos.columns).fillna("Sem setor")
    por_setor = pesos.T.groupby(setores).sum().T

    print(por_setor.mean().sort_values(ascending=False).round(3))
    print("maior setor na media das datas:", f"{por_setor.max(axis=1).mean():.0%}")
    print("maior setor em uma data:       ", f"{por_setor.max(axis=1).max():.0%}")

# sensibilidade aos pesos entre os blocos, sem custo
# serve para explicar o resultado; a estrategia oficial continua 60/40
teste_pesos = {}
for p in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    serie = p * retornos["Conservador"] + (1 - p) * retornos["Agressivo"]
    cagr, vol, sharpe, queda = medidas(serie)
    teste_pesos[f"{p:.0%}/{1 - p:.0%}"] = cagr
    print(f"{p:.0%}/{1 - p:.0%}  CAGR={cagr:6.2%} vol={vol:5.1%} Sharpe={sharpe:5.2f} queda={queda:6.1%}")

retornos.to_parquet("retornos_backtest.parquet")
pesos.to_parquet("carteiras_backtest.parquet")

# tabelas em csv para o painel de resultados (uma por pergunta)
os.makedirs("resultados", exist_ok=True)
chaves = {"Com custo": "estrategia", "Conservador": "conservador", "Agressivo": "agressivo",
          "Universo": "universo", "Ibovespa": "ibovespa", "Selic": "selic"}
colunas_csv = [c for c in chaves if c in retornos]

curva[colunas_csv].rename(columns=chaves).round(5).rename_axis("data").to_csv(
    "resultados/curva.csv", date_format="%Y-%m-%d")
anual[colunas_csv].rename(columns=chaves).round(5).rename_axis("ano").to_csv("resultados/anual.csv")

linhas = []
for c, nome in [("Com custo", "Estrategia com custo"), ("Estrategia", "Estrategia sem custo"),
                ("Conservador", "Conservador"), ("Agressivo", "Agressivo"),
                ("Universo", "Universo"), ("Ibovespa", "Ibovespa"), ("Selic", "Selic")]:
    if c not in retornos:
        continue
    cagr, vol, sharpe, queda = medidas(retornos[c])
    beta, alfa, ir = contra_ibov(retornos[c]) if tem_ibov and c != "Selic" else (np.nan,) * 3
    linhas.append({"serie": nome, "multiplo": curva[c].iloc[-1], "cagr": cagr, "vol": vol,
                   "sharpe": sharpe, "queda": queda, "beta": beta, "alfa": alfa, "ir": ir})
pd.DataFrame(linhas).round(4).to_csv("resultados/medidas.csv", index=False)

# fases do mercado, escolhidas pelos ciclos de juros e bolsa
fases = [("2010", "2013"), ("2014", "2016"), ("2017", "2019"), ("2020", "2022"), ("2023", str(retornos.index[-1].year))]
linhas = []
for ini, fim in fases:
    trecho = retornos.loc[ini:fim, colunas_csv]
    n = (trecho.index[-1] - trecho.index[0]).days / 365.25
    linha = (trecho.add(1).prod() ** (1 / n) - 1).rename(chaves).round(4).to_dict()
    linhas.append({"fase": f"{ini}-{fim}", **linha})
pd.DataFrame(linhas).to_csv("resultados/fases.csv", index=False)

if tem_ibov:
    explica.rename(columns={"Peso igual": "peso_igual", "Conservador": "conservador",
                            "Agressivo": "agressivo", "Custo": "custo", "Total": "total"}
                   ).round(5).rename_axis("ano").to_csv("resultados/explica.csv")
if tem_setores:
    por_setor.mean().sort_values(ascending=False).round(4).rename("peso").rename_axis("setor").to_csv(
        "resultados/setores.csv")

# Passo 6 - graficos
os.makedirs("graficos", exist_ok=True)
plt.rcParams.update({
    "figure.figsize": (11, 5.5), "figure.dpi": 130, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.8,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2,
})
cores = {"Com custo": "#2a78d6", "Ibovespa": "#eb6834", "Conservador": "#1baf7a",
         "Agressivo": "#eda100", "Selic": "#7a7974"}
nomes = {"Com custo": "Estrategia (com custo)"}
porcento = matplotlib.ticker.PercentFormatter(1, decimals=0)

def rotular_fim(ax, serie, cor, texto):
    ax.annotate(texto, (serie.index[-1], serie.iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color="#0b0b0b")
    ax.plot(serie.index[-1], serie.iloc[-1], "o", color=cor, ms=5)

# 6.1 curva acumulada, em escala log para que cada % pese igual em todo o periodo
fig, ax = plt.subplots()
for c in ["Selic", "Ibovespa", "Agressivo", "Conservador", "Com custo"]:
    if c not in curva:
        continue
    estilo = "--" if c == "Selic" else "-"
    ax.plot(curva.index, curva[c], estilo, color=cores[c], label=nomes.get(c, c),
            lw=2.4 if c == "Com custo" else 1.6)
    rotular_fim(ax, curva[c], cores[c], f"{curva[c].iloc[-1]:.1f}x")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_title("Quanto R$ 1 virou (escala log)")
ax.legend(loc="upper left")
ax.margins(x=0.06)
fig.tight_layout(); fig.savefig("graficos/1_curva_acumulada.png"); plt.close(fig)

# 6.2 quedas a partir do topo
fig, ax = plt.subplots(figsize=(11, 4))
for c in ["Ibovespa", "Com custo"]:
    if c in curva:
        dd = curva[c] / curva[c].cummax() - 1
        ax.plot(dd.index, dd, color=cores[c], label=nomes.get(c, c), lw=1.6)
ax.axhline(0, color="#b5b4ad", lw=1)
ax.yaxis.set_major_formatter(porcento)
ax.set_title("Queda a partir do topo (drawdown)")
ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig("graficos/2_drawdown.png"); plt.close(fig)

# 6.3 retorno de cada ano contra Ibov e Selic
colunas_ano = [c for c in ["Com custo", "Ibovespa", "Selic"] if c in anual]
fig, ax = plt.subplots()
largura = 0.8 / len(colunas_ano)
x = np.arange(len(anual))
for i, c in enumerate(colunas_ano):
    ax.bar(x + (i - (len(colunas_ano) - 1) / 2) * largura, anual[c], largura * 0.9,
           color=cores[c], label=nomes.get(c, c))
ax.axhline(0, color="#52514e", lw=1)
ax.set_xticks(x, [str(a) for a in anual.index])
ax.yaxis.set_major_formatter(porcento)
ax.set_title(f"Retorno por ano ({anual.index[-1]} ate {retornos.index[-1]:%d/%m})")
ax.legend(loc="upper left", ncol=len(colunas_ano))
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("graficos/3_retorno_anual.png"); plt.close(fig)

# 6.4 retorno de 12 meses moveis menos a Selic: quando valeu a pena correr o risco
movel = curva / curva.shift(252) - 1
fig, ax = plt.subplots(figsize=(11, 4))
for c in ["Ibovespa", "Com custo"]:
    if c in movel:
        exc = (movel[c] - movel["Selic"]).dropna()
        ax.plot(exc.index, exc, color=cores[c], label=nomes.get(c, c) + " menos Selic", lw=1.6)
ax.axhline(0, color="#52514e", lw=1)
ax.yaxis.set_major_formatter(porcento)
ax.set_title("Retorno em 12 meses acima da Selic")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("graficos/4_acima_da_selic_12m.png"); plt.close(fig)

# 6.5 teste dos pesos entre os blocos
cagr_selic = medidas(retornos["Selic"])[0]
fig, ax = plt.subplots(figsize=(8, 4))
barras = ax.bar(list(teste_pesos), list(teste_pesos.values()), 0.6, color="#2a78d6")
ax.bar_label(barras, labels=[f"{v:.1%}" for v in teste_pesos.values()], padding=3, fontsize=9)
ax.axhline(cagr_selic, color=cores["Selic"], ls="--", lw=1.5)
ax.annotate(f"Selic {cagr_selic:.1%}", (-0.4, cagr_selic), xytext=(0, 4),
            textcoords="offset points", fontsize=9)
ax.yaxis.set_major_formatter(porcento)
ax.set_title("CAGR por divisao Conservador/Agressivo (sem custo)")
ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("graficos/5_teste_pesos.png"); plt.close(fig)

# 6.6 peso medio por setor
if tem_setores:
    media = por_setor.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.barh(media.index, media, 0.6,
                     color=["#b5b4ad" if s == "Sem setor" else "#2a78d6" for s in media.index])
    ax.bar_label(barras, labels=[f"{v:.0%}" for v in media], padding=3, fontsize=9)
    ax.xaxis.set_major_formatter(porcento)
    ax.set_title("Peso medio por setor")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig("graficos/6_setores.png"); plt.close(fig)

# Passo 7 - relatorio completo do QuantStats (abre no navegador)
# o qs so aceita rf fixo: usa a Selic media do periodo, ao ano
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    qs.reports.html(retornos["Com custo"], benchmark=retornos["Ibovespa"] if tem_ibov else None,
                    rf=cagr_selic, title="Estrategia IQF vs Ibovespa",
                    output="relatorio_quantstats.html")
print("graficos em graficos/ e relatorio em relatorio_quantstats.html")
