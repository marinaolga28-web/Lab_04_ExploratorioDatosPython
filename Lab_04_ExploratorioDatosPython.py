# =====================================================
# IMPORTAR LIBRERÍAS
# =====================================================
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
 
# =====================================================
# CARGAR DATOS
# =====================================================
 
ventas = pd.read_csv("/workspace/ventas_peliculas.csv")
promocion = pd.read_csv("/workspace/promocion_peliculas.csv")
 
print("VENTAS")
print(ventas.head())
 
print("\nPROMOCIONES")
print(promocion.head())
 
# =====================================================
# COMBINAR DATAFRAMES
# =====================================================
 
df = pd.merge(
ventas,
promocion,
on=["fecha", "titulo"],
how="left",
validate="1:m"
)
 
# =====================================================
# IDENTIFICAR VALORES NULOS
# =====================================================
 
print("\nVALORES NULOS")
print(df.isna().sum())
 
# Rellenar nulos
 
df = df.ffill().bfill()
 
print("\nVALORES NULOS DESPUÉS DEL RELLENO")
print(df.isna().sum())
 
# =====================================================
# VENTAS SEMANALES (GRÁFICO DE LÍNEAS)
# =====================================================
 
plt.figure(figsize=(10,5))
 
plt.plot(
df["fecha"],
df["ventas_boletos"],
color="steelblue"
)
 
plt.title("Ventas semanales de boletos")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
 
plt.xticks(rotation=45)
 
plt.show()
 
# =====================================================
# COMPARACIÓN CON Y SIN PROMOCIÓN
# =====================================================
 
ventas_promedio = df.groupby(
"eventos_promocionales"
)["ventas_boletos"].mean()
 
ventas_promedio.plot(
kind="bar",
color=["gray", "steelblue"]
)
 
plt.title(
"Ventas promedio con y sin promoción"
)
 
plt.ylabel("Ventas promedio")
 
plt.show()
 
# =====================================================
# DETECCIÓN DE OUTLIERS
# =====================================================
 
df["ventas_boletos"].plot.box(
title="Boxplot de ventas de boletos"
)
 
plt.show()
 
# =====================================================
# CORRELACIÓN
# =====================================================
 
correlacion = df[
[
"ventas_boletos",
"calificacion",
"eventos_promocionales",
"participacion_promocion"
]
].corr()
 
print("\nMATRIZ DE CORRELACIÓN")
print(correlacion)
 
# =====================================================
# MODELO DE REGRESIÓN LINEAL
# =====================================================
 
X = df[
[
"calificacion",
"eventos_promocionales",
"participacion_promocion"
]
]
 
y = df["ventas_boletos"]
 
# Añadir constante
 
X = sm.add_constant(X)
 
modelo = sm.OLS(
y,
X
).fit()
 
# =====================================================
# RESUMEN DEL MODELO
# =====================================================
 
print(modelo.summary())
 
# =====================================================
# PREDICCIONES
# =====================================================
 
df["predicciones"] = modelo.predict(X)
 
# =====================================================
# GRÁFICO REAL VS PREDICHO
# =====================================================
 
plt.figure(figsize=(10,5))
 
plt.plot(
df.index,
df["ventas_boletos"],
label="Real",
color="steelblue"
)
 
plt.plot(
df.index,
df["predicciones"],
label="Predicción",
color="red"
)
 
plt.title(
"Ventas reales vs predichas"
)
 
plt.xlabel("Observación")
plt.ylabel("Ventas")
 
plt.legend()
 
plt.show()
