"""
Universidad Autonoma Metropolitana - Iztapalapa
Temas selectos de Bases de Datos
Practica 1: Data Cleaning
Por: Edgar Daniel Rodriguez Herrera
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import scipy.stats
from scipy import stats

""" Leemos todo el conjunto de datos del archivo nfl.csv """
nfl_data = pd.read_csv("./nfl.csv", low_memory=False)
# low_memory se usa para sobrepasar el error de mixed data types

"""semilla para reproducibilidad"""
np.random.seed(0)

"""mostramos los primeros datos"""
print(nfl_data.head())#regresa las primeras 5 filas si no se especifica un numero

"""Contamos los datos faltantes NaN"""
datos_faltantes= nfl_data.isnull().sum()
#isnull() regresa un boolean si una celda es NaN
#sum() regresa suma total despues de contar cada booleano y regresa en tabla de dato: num de datos faltantes
print("Datos faltantes:\n", datos_faltantes)

"""Calculamos el porcentaje de datos faltantes """
"""Para toda NFL Data"""
nfl_data_tamano= nfl_data.shape
#print("\nNFL Data\nFilas: ", nfl_data_tamano[0], "\nColumnas: ", nfl_data_tamano[1])
#shape regresa una tupla indicando el numero de filas y columnas en el Dataframe
num_celdas= np.prod(nfl_data.shape)
#print("Numero de celdas: ", num_celdas)
num_celdas_faltantes= datos_faltantes.sum() #regresa la suma total de los datos faltantes de cada dato
porcentaje_datos_faltantes= (num_celdas_faltantes/num_celdas)*100
print("\nPorcentaje de datos faltantes en NFL Data: ", porcentaje_datos_faltantes)
"""Para la columna yacWPA"""
print("Datos en columna yacWPA: ", nfl_data["yacWPA"].shape[0])
datos_faltantes_yacWPA= nfl_data["yacWPA"].isnull().sum()
print("Datos faltantes en la columna yacWPA: ", datos_faltantes_yacWPA)
porcentaje_dat_faltantes_yacWPA= (datos_faltantes_yacWPA/nfl_data["yacWPA"].shape[0])*100
print("Porcentaje de datos faltantes en la columna yacWPA: ", porcentaje_dat_faltantes_yacWPA)

"""
Por que faltan los datos faltantes?
Es prudente intentar estimar los datos faltantes?

A veces, se decide eliminar los datos faltantes
(aunque muchas veces no es recomendable por la posible perdida de informacion)

#una manera de borrar valores nulos es con la funcion dropna() que elimina el renglon si tiene almenos 1 valor nulo
nfl_drop= nfl_data.dropna(axis=1) #borra las columnas axis=1 que tengan al menos 1 valor nulo
print("\nNFL Data Drop\nFilas: ", nfl_drop.shape[0], "\nColumnas", nfl_drop.shape[1], "\n")

Otra opcion para lidiar con los datos faltantes es llenando dichos valores. """

#Obtenmos un subconjunto de datos
#nfl_subconjunto= nfl_data.loc[:, 'yacWPA'].head() #loc[row, column]
#print("\nColumna yacWPA original: \n",nfl_subconjunto)

"""Rellenamos valores nulos con 0"""
#print(nfl_subconjunto.fillna(0))

"""Rellenamos valores nulos con valor siguiente dentro de la misma columna (valor de abajo)"""
#print(nfl_subconjunto.bfill(axis=0).fillna(0))
#fillna(0) al final se usa para aquellos ultimos valores que no tienen valor siguiente
#nfl_subconjunto_rellenado= nfl_subconjunto.bfill(axis=0).fillna(0)
#print("\nColumna 'yacWPA' rellenada con bfill: \n", nfl_subconjunto_rellenado)
nfl_data["yacWPA"].bfill(axis=0).fillna(0)
print("\nColumna 'yacWPA' rellenada con bfill: \n", nfl_data["yacWPA"])

"""Generamos un resumen estadistica de la columna "yacWPA" """
#print("\nEstadisticas de la columna yacWPA: \n",nfl_data["yacWPA"].describe())
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
Q1= nfl_data["yacWPA"].quantile(0.25)
Q3= nfl_data["yacWPA"].quantile(0.75)
IQR= Q3-Q1
limite_superior= Q3+1.5*IQR
limite_inferior= Q1-1.5*IQR
outliers_iqr_superior= nfl_data[nfl_data["yacWPA"] > limite_superior]
outliers_iqr_inferior= nfl_data[nfl_data["yacWPA"] < limite_inferior]
print(f"\nNumero de outliers (IQR): {len(outliers_iqr_superior)+len(outliers_iqr_inferior)}")

"""Metodo Z-score"""
z_scores= np.abs(stats.zscore(nfl_data["yacWPA"]))
outliers_z_superior= nfl_data[nfl_data["yacWPA"] > 3]
outliers_z_inferior= nfl_data[nfl_data["yacWPA"] < -3]
print(f"Numero de outliers (z-score): {len(outliers_z_superior)+len(outliers_z_inferior)}")

"""Generamos un boxplot y un historgrama para observar los datos fuera de rango"""
plt.figure(figsize=(12,5))#(anchura, altura) en pulgadas
plt.subplot(1,2,1)#(1 fila, 2 columnas, 1a figura)
sns.boxplot(x=nfl_data["yacWPA"])
plt.title("Boxplot de yacWPA")

plt.subplot(1,2,2)#(1 fila, 2 columnas, 2a figura)
sns.histplot(nfl_data["yacWPA"], kde=True)
plt.title("Histograma de yacWPA")
#plt.show()

"""Eliminamos outliers"""
#Eliminamos outliers solo con IQR
df_sin_outliers_superiores= nfl_data[nfl_data["yacWPA"] <= limite_superior]
df_sin_outliers= df_sin_outliers_superiores[df_sin_outliers_superiores["yacWPA"] >= limite_inferior]

#Imputacion con la mediana
mediana= nfl_data["yacWPA"].median()
df_imputado= nfl_data.copy()
df_imputado["yacWPA"]= np.where(df_imputado["yacWPA"] > limite_superior, mediana, df_imputado["yacWPA"])
#si los valores son mayores que limite_superior, regresa mediana, sino regresa df_imputado["yacWPA"]
df_imputado["yacWPA"]= np.where(df_imputado["yacWPA"] < limite_inferior, mediana, df_imputado["yacWPA"])


#Transformacion logaritmica
df_transformado= nfl_data.copy()
df_transformado["yacWPA"]= np.log1p(df_transformado["yacWPA"])

"""Comparamos resultados antes y despues de la correccion de datos"""
print("\nOriginal:\n", nfl_data["yacWPA"].describe())
print("\nSin outliers:\n", df_sin_outliers["yacWPA"].describe())
print("\nImputado:\n", df_imputado["yacWPA"].describe())
print("\nTransformado:\n", df_transformado["yacWPA"].describe())

"""Boxplot despues de corregir conjunto de datos"""
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(x=df_sin_outliers["yacWPA"])
plt.title("Boxplot despues de eliminar outliers")

plt.subplot(1,2,2)#(1 fila, 2 columnas, 2a figura)
sns.histplot(df_sin_outliers["yacWPA"], kde=True)
plt.title("Histograma despues de eliminar outliers")
plt.show()
