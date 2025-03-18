import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import scipy.stats
from scipy import stats

""" Leemos todo el conjunto de datos del archivo creditcard.csv """
df= pd.read_csv("./creditcard.csv")

"""Generamos un resumen estadistica de la columna "Time" """
print(df["Time"].describe())

""" Identificamos los valores fuera de rango """
"""Metodo del rango intercuartilico (IQR)"""
Q1= df["Time"].quantile(0.25)
Q3= df["Time"].quantile(0.75)
IQR= Q3-Q1
limite_superior= Q3+1.5*IQR
outliers_iqr= df[df["Time"] > limite_superior]
print(f"Numero de outliers (IQR): {len(outliers_iqr)}")

"""Metodo Z-score"""
z_scores= np.abs(stats.zscore(df["Time"]))
outliers_z= df[df["Time"] > 3]
print(f"Numero de outliers (z-score): {len(outliers_z)}")

"""Generamos un boxplot y un historgrama para observar los datos fuera de rango"""
plt.figure(figsize=(12,5))#(anchura, altura) en pulgadas
plt.subplot(1,2,1)#(1 fila, 2 columnas, 1a figura)
sns.boxplot(x=df["Time"])
plt.title("Boxplot de Time")

plt.subplot(1,2,2)#(1 fila, 2 columnas, 2a figura)
sns.histplot(df["Time"], kde=True)
plt.title("Histograma de Time")
#plt.show()

"""Eliminamos outliers"""
#Eliminamos outliers solo con IQR
df_sin_outliers= df[df["Time"] <= limite_superior]

#Imputacion con la mediana
mediana= df["Time"].median()
df_imputado= df.copy()
df_imputado["Time"]= np.where(df_imputado["Time"] > limite_superior, mediana, df_imputado["Time"])

#Transformacion logaritmica
df_transformado= df.copy()
df_transformado["Time"]= np.log1p(df_transformado["Time"])

"""Comparamos resultados antes y despues de la correccion de datos"""
print("Original:\n", df["Time"].describe())
print("\nSin outliers:\n", df_sin_outliers["Time"].describe())
print("\nImputado:\n", df_imputado["Time"].describe())
print("\nTransformado:\n", df_transformado["Time"].describe())

"""Boxplot despues de corregir conjunto de datos"""
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(x=df_sin_outliers["Time"])
plt.title("Boxplot despues de eliminar outliers")

plt.subplot(1,2,2)#(1 fila, 2 columnas, 2a figura)
sns.histplot(df_sin_outliers["Time"], kde=True)
plt.title("Histograma de Time")
plt.show()