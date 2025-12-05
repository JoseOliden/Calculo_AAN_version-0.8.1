import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
import os
import re
import io
import json
from datetime import datetime
from scipy.optimize import root
import sympy as sp
import matplotlib.pyplot as plt
import tempfile
import base64
# ------------------ RPT ---------------------------------

def limpiar(valor):
    if isinstance(valor, str):
        # elimina todos los espacios vacíos al inicio y al final
        valor = valor.strip()
    return valor

def procesar_RPT(rpt_file):
    if rpt_file is None:
        return None

    # 1. Leer directo desde uploader sin guardarlo
    contenido = rpt_file.read().decode("utf-8", errors="ignore")

    # 2. Convertir a pandas como una fila por línea
    lineas = contenido.splitlines()
    df = pd.DataFrame(lineas, columns=["linea"])

    # 3. Eliminar primeras 19 filas
    df = df.iloc[19:].reset_index(drop=True)

    # 4. Crear columna auxiliar sin espacios
    df["sin_espacios"] = df["linea"].str.lstrip()

    # 5. Filtrar patrones a eliminar
    patrones_excluir = ("Peak", "M =", "m =", "F =", "Error")
    df = df[~df["sin_espacios"].str.startswith(patrones_excluir)]

    # 6. Eliminar líneas que empiezan con exactamente 3 espacios
    df = df[~df["linea"].str.startswith("   ")]

    # 7. Eliminar líneas vacías o solo con espacios
    df = df[~df["linea"].str.strip().eq("")]

    # 8. Eliminar columna auxiliar
    df = df.drop(columns=["sin_espacios"])

    # 9. Reiniciar índices
    df = df.reset_index(drop=True)

    # 10. Quitar 1 espacio inicial (si hay) y todos los finales
    df = df.applymap(limpiar)

    # 11. Separar columnas y asignar nombres
    
    # primera columna
    df_tipo = df["linea"].str[:1].to_frame()
    df_tipo = df_tipo.applymap(limpiar)
    df_tipo.columns = ["Tipo"]
    
    # demas columnas
    df_demas = df["linea"].str[1:].to_frame()
    df_demas = df_demas.applymap(limpiar)
    df_demas_tab = df_demas["linea"].str.split(r"\s+", n=9, expand=True)
    df_demas_tab.columns = ["Peak No.", "ROI Start", "ROI End", "Peak Centroid",
        "Energy (keV)", "Net Peak Area", "Net Peak Uncert",
        "Continuum Counts", "Tentative Nuclide"
    ]
    
    # 12. Unir df
    df_tab = pd.concat([df_tipo, df_demas_tab], axis=1)
    st.success("Archivo procesado correctamente 🚀")
    return df_tab

def Selecion_Nucleidos_muestra(df_rpt_muestras,df_Nucleidos, df_database):
    df_filtrado = df[(df[""] == "P") & ((df["Energía"] > 100) | (df["Energía"] < 20))]
    return 0
def Selecion_Nucleidos_Au(df_rpt_Au,df_Nucleidos, df_database):
    # buscar en database energía de Au
    En_Au = float(411.8) 
    tol_Au = float(1.0)
    df_rpt_Au["Energy (keV)"] = pd.to_numeric(df_rpt_Au["Energy (keV)"], errors="coerce")
    df_energy_Au = df_rpt_Au[(df_rpt_Au["Tentative Nuclide"] == "AU-198") & ((df_rpt_Au["Energy (keV)"] > En_Au + tol_Au) | (df_rpt_Au["Energy (keV)"] < En_Au - tol_Au))]
    return df_energy_Au   
def Selecion_Nucleidos(df_rpt_Au,df_Nucleidos, df_databas):
    df_filtrado = df[(df[""] == "P") & ((df["Energía"] > 100) | (df["Energía"] < 20))]
    return 0   

# ------------------ kos ---------------------------------

def extraer_DATE_MEA_MEAS_TIM(k0s_file):
    if k0s_file is None:
        return None

    contenido = k0s_file.getvalue().decode("utf-8", errors="ignore")
    lineas = contenido.splitlines()

    fecha = hora = tiempo_real = tiempo_vivo = None

    for i, linea in enumerate(lineas):
        l = linea.strip()

        # -------------------- DATE_MEA --------------------
        if l.startswith("$DATE_MEA"):
            if i + 1 < len(lineas):
                datos = lineas[i + 1].strip().split()
                if len(datos) >= 2:
                    fecha = datos[0]
                    hora = datos[1]

        # -------------------- MEAS_TIM --------------------
        if l.startswith("$MEAS_TIM"):
            if i + 1 < len(lineas):
                datos = lineas[i + 1].strip().split()
                if len(datos) >= 2:
                    tiempo_vivo = datos[0]
                    tiempo_real = datos[1]
                    
    return fecha, hora, tiempo_vivo, tiempo_real
