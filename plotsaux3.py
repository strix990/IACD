##### For cleaning the Data


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotsaux2 as aux

def repair_row(row):
    peso, altura, imc = row["Peso"], row["Altura"], row["IMC"]
    missing = sum(pd.isna([peso, altura, imc]))

    if missing >= 2:
        return row

    if pd.isna(peso):
        row["Peso"] = imc * (altura ** 2)
    elif pd.isna(altura) and imc > 0:
        row["Altura"] = np.sqrt(peso / imc)
    elif pd.isna(imc):
        row["IMC"] = peso / (altura ** 2)

    return row


df = pd.read_excel("./DATASET.xls")

# IDADE  ---  (2 < idade < 19)
df["IDADE"] = pd.to_numeric(df["IDADE"], errors="coerce")
df = df[df["IDADE"].between(2, 19)]

numeric_cols = ["Peso", "Altura", "IMC", "PA SISTOLICA", "PA DIASTOLICA"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# PESO  ---  (8 < peso < 100)
df.loc[df["Peso"] <= 0, "Peso"] = np.nan

# ALTURA  ---  ajustes + conversão + limites (60cm < altura < 220cm)
df.loc[df["Altura"] == 0, "Altura"] = np.nan
df.loc[df["Altura"] < 3, "Altura"] = df["Altura"]           # já está em metros
df.loc[df["Altura"] > 100, "Altura"] = df["Altura"] / 100  # estava em cm

df = df.apply(repair_row, axis=1)

# ajustar novamente após repair
df.loc[df["Altura"] == 0, "Altura"] = np.nan
df.loc[df["Altura"] < 3, "Altura"] = df["Altura"]
df.loc[df["Altura"] > 100, "Altura"] = df["Altura"] / 100

# aplicar limites finais
df.loc[~df["Peso"].between(8, 100), "Peso"] = np.nan
df.loc[~df["Altura"].between(0.60, 2.20), "Altura"] = np.nan
df.loc[~df["IMC"].between(10, 60), "IMC"] = np.nan

# PA SISTOLICA  --- 50 < PAS < 250
df["PA SISTOLICA"] = df["PA SISTOLICA"].clip(lower=50, upper=250)

# PA DIASTOLICA  --- 30 < PAD < 150
df["PA DIASTOLICA"] = df["PA DIASTOLICA"].clip(lower=30, upper=150)

odf = df

aux.SpreadMeasure()
##DensityPlot()
##BoxPlot()
##Histogram()


output_path = "dataset_limpo.xlsx"
df.to_excel(output_path, index=False)

print(f"Arquivo Excel criado com sucesso: {output_path}")