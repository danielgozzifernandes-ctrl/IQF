import pandas as pd

MESES = {"Jan": 1, "Fev": 2, "Mar": 3, "Abr": 4, "Mai": 5, "Jun": 6,
         "Jul": 7, "Ago": 8, "Set": 9, "Out": 10, "Nov": 11, "Dez": 12}


def limpar(arquivo, texto):
    # 1. abrir
    bruto = pd.read_excel(arquivo)
    tabela = bruto.copy()

    # 2. cabecalho certo
    tabela.columns = tabela.iloc[2]
    tabela = tabela.drop(tabela.index[:3])
    tabela.columns.name = None

    # 3. nome das colunas
    tabela.columns = tabela.columns.str.replace(texto, "", regex=False)
    return tabela


def conferir(tabela):
    print("tamanho: ", tabela.shape)
    print("tipos:   ", set(tabela.dtypes.astype(str)))
    print("faltando:", int(tabela.isna().sum().sum()))
    print("periodo: ", tabela.index.min().date(), "a", tabela.index.max().date())
    print("vazias:  ", int((tabela.notna().sum() == 0).sum()), "acoes")


# precos e ROE: data no indice, texto para numero, tirar vazias e salvar
# o ROE fica datado pelo fim do trimestre do balanco, nao pela divulgacao
for entrada, texto, saida in [
    ("fechamento.xlsx", "Fechamento\najust p/ prov\nEm moeda orig\n", "precos_limpo.parquet"),
    ("ROE.xlsx", " RenPat(in)\n Em moeda orig\n de 3 meses\n consolid:sim*\n", "roe_limpo.parquet"),
]:
    tabela = limpar(entrada, texto)

    # 4. data no indice
    tabela["Data"] = pd.to_datetime(tabela["Data"])
    tabela = tabela.set_index("Data")

    # 5. texto para numero
    tabela = tabela.apply(pd.to_numeric, errors="coerce")
    conferir(tabela)

    # 6. tirar colunas vazias e salvar
    tabela = tabela.dropna(axis=1, how="all")
    tabela.to_parquet(saida)

# market cap e ineg: "Jan-2008" vira 31/01/2008, para so serem usados depois do mes fechar
for entrada, texto, saida in [
    ("marketcap.xlsx", "Valor Mercado\nda empresa\nEm moeda orig\nem milhares\n", "mktcap_limpo.parquet"),
    ("ineg.xlsx", "Negociabilidade\n1 meses\nEm moeda orig\n", "ineg_limpo.parquet"),
]:
    tabela = limpar(entrada, texto)
    partes = tabela["Data"].str.split("-", expand=True)
    datas = pd.to_datetime(dict(year=partes[1].astype(int), month=partes[0].map(MESES), day=1))
    tabela["Data"] = datas + pd.offsets.MonthEnd(0)
    tabela = tabela.set_index("Data")
    tabela = tabela.apply(pd.to_numeric, errors="coerce")
    conferir(tabela)
    tabela = tabela.dropna(axis=1, how="all")
    tabela.to_parquet(saida)

# ibov e setores: vieram do Yahoo Finance e ja estao limpos
ibov = pd.read_excel("ibov_yahoo.xlsx", index_col="Data")
print("ibov:    ", ibov.shape, ibov.index.min().date(), "a", ibov.index.max().date())
ibov.to_parquet("ibov.parquet")

setores = pd.read_excel("setores_yahoo.xlsx", index_col="Ticker")
print("setores: ", setores.shape, "sem setor:", int(setores["Setor"].isna().sum()))
setores.to_parquet("setores.parquet")
