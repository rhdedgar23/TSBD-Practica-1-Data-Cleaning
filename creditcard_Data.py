import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import scipy.stats
from scipy import stats

""" Leemos todo el conjunto de datos del archivo creditcard.csv """
df= pd.read_csv("./creditcard.csv")

"""Generamos un resumen estadistica de la columna "Amount" """
print(df["Amount"].describe())
#df[] los corchetes indican que queremos que la informacion se presente en forma de tabla
"""Las estadisticas mostradas son:
    count: numero de valores no nulos de la columna
    mean: media aritmetica de los valores
    std: desviacion estandar, que mide la dispersion de los datos
    min: valor minimo de la columna
    25%: primer cuartil (Q1). Es el valor que deja al 25% de los dtos por debajo y el 75% encima. 
    50%: segundo cuartil (Q2). Es el valor medio de los datos (mediana). 
        Divide la distribucion en partes iguales, dejando al 50% de los datos a cada lado. 
    75%: tercer cuartil (Q3). Deja el 75% de los datos por debajo y el 25%  por encima. 
    max: valor maximo de la columna. """

""" Identificamos los valores fuera de rango """
"""Metodo del rango intercuartilico (IQR)"""
Q1= df["Amount"].quantile(0.25)
Q3= df["Amount"].quantile(0.75)
IQR= Q3-Q1
limite_superior= Q3+1.5*IQR
outliers_iqr= df[df["Amount"] > limite_superior]
print(f"Numero de outliers (IQR): {len(outliers_iqr)}")

"""Metodo Z-score"""
z_scores= np.abs(stats.zscore(df["Amount"]))
outliers_z= df[df["Amount"] > 3]
print(f"Numero de outliers (z-score): {len(outliers_z)}")

"""Generamos un boxplot y un historgrama para observar los datos fuera de rango"""
plt.figure(figsize=(12,5))#(anchura, altura) en pulgadas
plt.subplot(1,2,1)#(1 fila, 2 columnas, 1a figura)
sns.boxplot(x=df["Amount"])
plt.title("Boxplot de Amount")

plt.subplot(1,2,2)#(1 fila, 2 columnas, 2a figura)
sns.histplot(df["Amount"], kde=True)
plt.title("Histograma de Amount")
#plt.show()

"""Eliminamos outliers"""
#Eliminamos outliers solo con IQR
df_sin_outliers= df[df["Amount"] <= limite_superior]

#Imputacion con la mediana
mediana= df["Amount"].median()
df_imputado= df.copy()
df_imputado["Amount"]= np.where(df_imputado["Amount"] > limite_superior, mediana, df_imputado["Amount"])

#Transformacion logaritmica
df_transformado= df.copy()
df_transformado["Amount"]= np.log1p(df_transformado["Amount"])

"""Comparamos resultados antes y despues de la correccion de datos"""
print("Original:\n", df["Amount"].describe())
print("\nSin outliers:\n", df_sin_outliers["Amount"].describe())
print("\nImputado:\n", df_imputado["Amount"].describe())
print("\nTransformado:\n", df_transformado["Amount"].describe())

"""Boxplot despues de corregir conjunto de datos"""
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(x=df_sin_outliers["Amount"])
plt.title("Boxplot despues de eliminar outliers")

plt.subplot(1,2,2)#(1 fila, 2 columnas, 2a figura)
sns.histplot(df_sin_outliers["Amount"], kde=True)
plt.title("Histograma de Amount")
plt.show()