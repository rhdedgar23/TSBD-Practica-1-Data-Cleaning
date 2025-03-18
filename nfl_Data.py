"""
Universidad Autonoma Metropolitana - Iztapalapa
Temas selectos de Bases de Datos
Practica 1: Data Cleaning
Por: Edgar Daniel Rodriguez Herrera
"""

import pandas as pd
import numpy as np

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
nfl_data_tamano= nfl_data.shape
print("\nNFL Data\nFilas: ", nfl_data_tamano[0], "\nColumnas: ", nfl_data_tamano[1])
#shape regresa una tupla indicando el numero de filas y columnas en el Dataframe
num_celdas= np.prod(nfl_data.shape)
print("Numero de celdas: ", num_celdas)
num_celdas_faltantes= datos_faltantes.sum() #regresa la suma total de los datos faltantes de cada dato
porcentaje_datos_faltantes= (num_celdas_faltantes/num_celdas)*100
print("Porcentaje de datos faltantes: ", porcentaje_datos_faltantes)

"""
Por que faltan los datos faltantes?
Es prudente intentar estimar los datos faltantes?
"""

"""
A veces, se decide eliminar los datos faltantes
(aunque muchas veces no es recomendable por la posible perdida de informacion)
"""
#una manera de borrar valores nulos es con la funcion dropna() que elimina el renglon si tiene almenos 1 valor nulo
nfl_drop= nfl_data.dropna(axis=1) #borra las columnas axis=1 que tengan al menos 1 valor nulo
print("\nNFL Data Drop\nFilas: ", nfl_drop.shape[0], "\nColumnas", nfl_drop.shape[1], "\n")

"""
Otra opcion para lidiar con los datos faltantes es llenando dichos valores. 
"""
#Obtenmos un subconjunto de datos
nfl_subconjunto= nfl_data.loc[:, 'EPA':'Season'].head() #loc[row, column]
print(nfl_subconjunto)

"""Rellenamos valores nulos con 0"""
print(nfl_subconjunto.fillna(0))

"""Rellenamos valores nulos con valor siguiente dentro de la misma columna (valor de abajo)"""
print(nfl_subconjunto.bfill(axis=0).fillna(0))
#fillna(0) al final se usa para aquellos ultimos valores que no tienen valor siguiente

