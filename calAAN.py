import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
import os
import re
import io
import json
from datetime import datetime, timedelta
from scipy.optimize import root
import sympy as sp
import matplotlib.pyplot as plt
import tempfile
import base64
# ==============================================================================
#                        FUNCIONES AUXILIARES CALCULO
# ==============================================================================

# ----------------------------- Archivos Aesp ---------------------------------#

def Aesp(Cn_i, w_i,lam,tr,td,ti,tv,e):
  # Calcula la actividad específica
  C_i = float(lam/(1-np.exp(-lam*tr)))
  D_i = float(np.exp(lam*td))
  H_i = float(tr/tv)
  S_i = float(1-np.exp(-lam*ti))
  return Cn_i*D_i*C_i*H_i/(S_i*w_i) # se agregó e

# ---------------------------- Calculo de alfa --------------------------------#
def crear_df_comparadores():
  # comparadores [Au Co Mo]
  #Q0_c = np.array([15.712, 2.041, 53.1])    #
  df = pd.DataFrame(
      data=[
        [1.00,      0.02272,  52346, 0.0000001358, 15.712, 0.000002977,    5.65,  306314, 1500, 10800, 1478.0],
        [1.32,      0.0113,   36082, 0.000051573,  2.041,  0.000000004167, 136.0, 306314, 1500, 10800, 1478.0],
        [0.0000846, 0.016053, 39272, 0.002088591,  50.365, 0.00000291994,  241,   299161, 900,  10800, 866.0]
      ],
      columns=["k0", "efe", "Cn", "w", "Q0", "lambda", "Er", "t_dec", "t_real", "t_irr", "t_vivo" ],
      index=["Au", "Co", "Mo"]
    )
  return df



def cal_alfa(df_comp):
  def equations(vars, *par):
    # Define el sistema de ecuaciones para hallar alfa
    alfa = vars[0]
    Aesp_1,k0_1,e_1,Er_1,Q0_1,Aesp_2,k0_2,e_2,Er_2,Q0_2, Aesp_3,k0_3,e_3,Er_3,Q0_3 = par
    eq1 = ((1-(Aesp_2/Aesp_1)*(k0_1/k0_2)*(e_1/e_2))**(-1) - (1-(Aesp_3/Aesp_1)*(k0_1/k0_3)*(e_1/e_3))**(-1))*(Q0_1 - 0.429)/(Er_1**(alfa)) - ((1-(Aesp_2/Aesp_1)*(k0_1/k0_2)*(e_1/e_2))**(-1))*(Q0_2 - 0.429)/(Er_2**(alfa)) + ((1-(Aesp_3/Aesp_1)*(k0_1/k0_3)*(e_1/e_3))**(-1))*(Q0_3 - 0.429)/(Er_3**(alfa))
    return [eq1]
  
  # calcula alfa
  #k0_c, e_c, Q0_c, Cn_c, w_c, lam_c,Er_c, td_c, tr_c, ti_c, tv_c = par_comp
  k0_c = df_comp["k0"].to_numpy()
  e_c = df_comp["efe"].to_numpy()
  Q0_c = df_comp["Q0"].to_numpy()
  Cn_c = df_comp["Cn"].to_numpy()
  w_c = df_comp["w"].to_numpy()
  lam_c = df_comp["lambda"].to_numpy()
  Er_c = df_comp["Er"].to_numpy()
  td_c = df_comp["t_dec"].to_numpy()
  tr_c = df_comp["t_real"].to_numpy()
  ti_c = df_comp["t_irr"].to_numpy()
  tv_c = df_comp["t_vivo"].to_numpy()

  Aesp_c = np.zeros(len(k0_c))
  for i in range(len(k0_c)):
    Aesp_c[i] = Aesp(Cn_c[i], w_c[i],lam_c[i],tr_c[i],td_c[i],ti_c[i],tv_c[i],e_c[i]) # Activida especifica del elemento comparador
  initial_guesses = [0.0]
  par = (Aesp_c[0], k0_c[0], e_c[0], Er_c[0], Q0_c[0], Aesp_c[1], k0_c[1], e_c[1], Er_c[1], Q0_c[1], Aesp_c[2], k0_c[2], e_c[2], Er_c[2], Q0_c[2])
  solution = root(equations, initial_guesses, args = par)
  alfa = solution.x

  # Calcular f
  Q0_alfa_c = np.zeros(len(k0_c))
  for i in range(len(k0_c)):
    Q0_alfa_c[i] = cal_Q0_alfa_i(Q0_c[i],Er_c[i],alfa)
  f = cal_f_alfa(Q0_alfa_c,Aesp_c,e_c,k0_c)
  
  return alfa[0], f
# ---------------------------- Calculo de f --------------------------------#
def cal_f_alfa(Q0_alfa_c,Aesp_c,e_c,k0_c):
  # calcula f
  return ((k0_c[0]/k0_c[1])*(e_c[0]/e_c[1])*Q0_alfa_c[0]  - (Aesp_c[0]/Aesp_c[1])*Q0_alfa_c[1])/(Aesp_c[0]/Aesp_c[1] - (k0_c[0]/k0_c[1])*(e_c[0]/e_c[1]))
# ------------------------ Calculo de concentración ---------------------------#

def cal_Q0_alfa_i(Q0,Er,alfa):
  # calcula Q0_alfa del elemento i
  #(Q0-0.429)/(Er**alfa) + 0.429/((2*alfa+1)*0.55**alfa) literatura
  # (Q0-0.429)/(Er**alfa) + 0.429/(2*alfa+0.55**alfa) # MACROS
  return (Q0-0.429)/(Er**alfa) + 0.429/(2*alfa+0.55**alfa)

def conc(df_muestra, w,td_i,ti_i,tv_i,tr_i, df_comp_Au, w_Au,td_c_Au,ti_c_Au,tv_c_Au,tr_c_Au, alfa, f, geometria):
  alfa = 0.226 # Forzar valor de alfa 
  f = 34       # Forzar valor de f
  
  # Comparador Au
  #k0_c_Au, e_c_Au, Q0_c_Au, Cn_c_Au, w_c_Au, lam_c_Au, Er_c_Au, td_c_Au, tr_c_Au, ti_c_Au, tv_c_Au =  par_comp_Au
  k0_c_Au = df_comp_Au["K0"].to_numpy()
  if geometria == "50 mm":
    e_c_Au = df_comp_Au["EFIGAMMA50"].to_numpy()*df_comp_Au["COI ROSSBACH"].to_numpy()
  if geometria == "185 mm":
    e_c_Au = df_comp_Au["EFIGAMMA185"].to_numpy()*df_comp_Au["COI GAMMA185"].to_numpy()
  Q0_c_Au = df_comp_Au["Q0"].to_numpy()
  Cn_c_Au = df_comp_Au["Net Peak Area"].to_numpy()
  w_c_Au = w_Au
  lam_c_Au = np.log(2)/df_comp_Au["t(1/2) s"].to_numpy()
  Er_c_Au = df_comp_Au["EREF"].to_numpy()

  Aesp_c_Au = Aesp(float(Cn_c_Au[0]), w_c_Au, float(lam_c_Au[0]), tr_c_Au, td_c_Au, ti_c_Au, tv_c_Au, float(e_c_Au[0]))
  Q0_alfa_c_Au = cal_Q0_alfa_i(Q0_c_Au[0],Er_c_Au[0],alfa)
  
  # muestra
  #k0_i, e_i, Q0_i, Cn_i, w_i, lamb_i, Er_i, td_i, tr_i, ti_i, tv_i = par_ele
  k0_i = df_muestra["K0"].to_numpy()
  if geometria == "50 mm":
    e_i = df_muestra["EFIGAMMA50"].to_numpy()*df_muestra["COI ROSSBACH"].to_numpy()
    #e_i = df_muestra["efe"].to_numpy()
  if geometria == "185 mm":
    e_i = df_muestra["EFIGAMMA185"].to_numpy()*df_muestra["COI GAMMA185"].to_numpy()
  Q0_i = df_muestra["Q0"].to_numpy()
  Cn_i = df_muestra["Net Peak Area"].to_numpy()
  w_i = w
  lam_i = np.log(2)/df_muestra["t(1/2) s"].to_numpy()
  Er_i = df_muestra["EREF"].to_numpy()

  Aesp_i = np.zeros(len(k0_i))
  Q0_alfa_i = np.zeros(len(k0_i))
  for i in range(len(k0_i)):
    Aesp_i = Aesp(float(Cn_i[i]), w_i, float(lam_i[i]), tr_i, td_i, ti_i, tv_i, float(e_i[i]))
    Q0_alfa_i = cal_Q0_alfa_i(Q0_i[i],Er_i[i],alfa)

  C = np.zeros(len(k0_i))
  for i in range(len(k0_i)):
    # Calcula la concentración del elemento i en la muestra
    C[i] = (Aesp_i[i]/Aesp_c_Au)*(k0_c_Au/k0_i[i])*(e_c_Au/e_i[i])*((f + Q0_alfa_c_Au)/(f + Q0_alfa_i[i]))
  
  return C 

# ------------------------ Calculo de Incertidumbre ---------------------------#

def cal_U(Val_ini,u_v_ini):
  (Cn, Cn_1, Cn_2, Cn_c_Au, Er, Er_1, Er_2, Er_c_Au, Q0, Q0_1, Q0_2, Q0_c_Au,
   alfa, e, e_1, e_2, e_c_Au, k0, k0_1, k0_2, k0_c_Au, lamb, lamb_1, lamb_2,
   lamb_c_Au, td, td_1, td_2, td_c_Au, ti, ti_1, ti_2, ti_c_Au, tr, tr_1, tr_2,
   tr_c_Au, tv, tv_1, tv_2, tv_c_Au, w, w_1, w_2, w_c_Au) = Val_ini
  (u_Cn, u_Cn_1, u_Cn_2,u_Cn_c_Au, u_Er, u_Er_1, u_Er_2, u_Er_c_Au, u_Q0, u_Q0_1,
   u_Q0_2, u_Q0_c_Au, u_alfa, u_e, u_e_1, u_e_2, u_e_c_Au, u_k0, u_k0_1, u_k0_2,
   u_k0_c_Au, u_lamb, u_lamb_1, u_lamb_2, u_lamb_c_Au, u_td, u_td_1, u_td_2,
   u_td_c_Au, u_ti, u_ti_1, u_ti_2, u_ti_c_Au, u_tr, u_tr_1, u_tr_2, u_tr_c_Au,
   u_tv, u_tv_1, u_tv_2, u_tv_c_Au, u_w, u_w_1, u_w_2, u_w_c_Au) = u_v_ini

  # Aesp
  # [Cn, lamb, td, ti, tr, tv, w]
  Val_ini_Aesp = (Cn, lamb, td, ti, tr, tv, w)
  u_v_ini_Aesp = (u_Cn, u_lamb, u_td, u_ti, u_tr, u_tv, u_w)
  u_Aesp, Aesp  = cal_U_Aesp(Val_ini_Aesp,u_v_ini_Aesp)


  # Aesp_1
  # [Cn, lamb, td, ti, tr, tv, w]
  Val_ini_Aesp_1 = (Cn_1, lamb_1, td_1, ti_1, tr_1, tv_1, w_1)
  u_v_ini_Aesp_1 = (u_Cn_1, u_lamb_1, u_td_1, u_ti_1, u_tr_1, u_tv_1, u_w_1)
  u_Aesp_1, Aesp_1  = cal_U_Aesp(Val_ini_Aesp_1,u_v_ini_Aesp_1)


  # Aesp_2
  # [Cn, lamb, td, ti, tr, tv, w]
  Val_ini_Aesp_2 = (Cn_2, lamb_2, td_2, ti_2, tr_2, tv_2, w_2)
  u_v_ini_Aesp_2 = (u_Cn_2, u_lamb_2, u_td_2, u_ti_2, u_tr_2, u_tv_2, u_w_2)
  u_Aesp_2, Aesp_2  = cal_U_Aesp(Val_ini_Aesp_2,u_v_ini_Aesp_2)


  # Aesp_c_Au
  # [Cn_c_Au, lamb_c_Au, td_c_Au, ti_c_Au, tr_c_Au, tv_c_Au, w_c_Au]
  Val_ini_Aesp_c_Au = (Cn_c_Au, lamb_c_Au, td_c_Au, ti_c_Au, tr_c_Au, tv_c_Au,
                       w_c_Au)
  u_v_ini_Aesp_c_Au = (u_Cn_c_Au, u_lamb_c_Au, u_td_c_Au, u_ti_c_Au, u_tr_c_Au,
                       u_tv_c_Au, u_w_c_Au)
  u_Aesp_c_Au, Aesp_c_Au  = cal_U_Aesp(Val_ini_Aesp_c_Au,u_v_ini_Aesp_c_Au)


  # [Aesp, Aesp_1, Aesp_2, Aesp_c_Au, Er, Er_1, Er_2, Er_c_Au, Q0, Q0_1, Q0_2, Q0_c_Au, alpha, e, e_1, e_2, e_c_Au, k0, k0_1, k0_2, k0_c_Au]
  Val_ini_con = (Aesp, Aesp_1, Aesp_2, Aesp_c_Au, Er, Er_1, Er_2, Er_c_Au, Q0,
                 Q0_1, Q0_2, Q0_c_Au, alfa, e, e_1, e_2, e_c_Au, k0, k0_1, k0_2,
                 k0_c_Au)
  u_v_ini_con = (u_Aesp, u_Aesp_1, u_Aesp_2, u_Aesp_c_Au, u_Er, u_Er_1, u_Er_2,
                 u_Er_c_Au, u_Q0, u_Q0_1, u_Q0_2, u_Q0_c_Au, u_alfa, u_e, u_e_1,
                 u_e_2, u_e_c_Au, u_k0, u_k0_1, u_k0_2, u_k0_c_Au)
  # calcula incertidumbre
  #[Aesp, Aesp_1, Q0_alfa, Q0_alfa_1, e, e_1, f, k0, k0_1]
  formula_str = "(Aesp/Aesp_c_Au)*(k0_c_Au/k0)*(e_c_Au/e)*(((k0_1/k0_2)*(e_1/e_2)*((Q0_1 -0.429)/((Er_1)**alfa)+0.429/((2*alfa-1)*0.55**alfa))-(Aesp_1/Aesp_2)*((Q0_2 -0.429)/((Er_2)**alfa)+0.429/((2*alfa-1)*0.55**alfa)))/((Aesp_1/Aesp_2)-(k0_1/k0_2)*(e_1/e_2))+((Q0_c_Au -0.429)/((Er_c_Au)**alfa)+0.429/((2*alfa-1)*0.55**alfa))) / (((k0_1/k0_2)*(e_1/e_2)*((Q0_1 -0.429)/((Er_1)**alfa)+0.429/((2*alfa-1)*0.55**alfa))-(Aesp_1/Aesp_2)*((Q0_2 -0.429)/((Er_2)**alfa)+0.429/((2*alfa-1)*0.55**alfa)))/((Aesp_1/Aesp_2)-(k0_1/k0_2)*(e_1/e_2))+((Q0 -0.429)/((Er)**alfa)+0.429/((2*alfa-1)*0.55**alfa)))"
  derivadas = cal_derivadas(Val_ini_con)

  #print(Val_ini)
  # Extraer variables únicas
  try:
      variables = sorted(list(sp.sympify(formula_str).free_symbols), key=lambda x: str(x))
  except Exception as e:
      st.error(f"Error al interpretar la fórmula: {e}")
  # Entrada de valores e incertidumbres
  valores = {}
  incertidumbres = {}
  i = 0
  for var in variables:
    valor =  Val_ini_con[i]
    incertidumbre = u_v_ini_con[i]
    valores[str(var)] = valor
    incertidumbres[str(var)] = incertidumbre
    i = i + 1

  # Cálculo
  try:
    # Definir símbolos
    simbolos = {str(v): sp.Symbol(str(v)) for v in variables}
    formula_sym = sp.sympify(formula_str)
    # Calcular valor central
    y_val = float(formula_sym.evalf(subs=valores))
    u_y_squared = 0
    contribuciones = []
    kkk = 0
    for v in variables:
        var_name = str(v)
        sensibilidad = derivadas[kkk]
        kkk = kkk + 1
        u_i = incertidumbres[var_name]
        contrib = (float(sensibilidad) * u_i)**2
        u_rel_i = u_i / valores[var_name] if valores[var_name] != 0 else np.nan
        u_y_squared += contrib
    if isinstance(u_y_squared, np.ndarray):
      u_y_squared = u_y_squared.item()
    if isinstance(u_y_squared, list):
      u_y_squared = u_y_squared[0]

    u_y = np.sqrt(u_y_squared) # incertidumbre combinada
    u_rel_y = u_y / y_val if y_val != 0 else np.nan

    # Calcular porcentaje de contribución
    #for c in contribuciones:
    #    c["% Contribución"] = 100 * c["Contribución a u(y)²"] / u_y_squared if u_y_squared > 0 else np.nan
  except Exception as e:
      st.error(f"Ocurrió un error en el cálculo: {e}")
  return u_y, y_val

def cal_U_Aesp(Val_ini,u_v_ini):
  # [Cn, lamb, t_d, ti, tr, tv, w]
  # calcula incertidumbre
  formula_str = "(Cn*exp(lamb*td)*lamb*tr)/((1-exp(-lamb*ti))*(1-exp(-lamb*tr))*w*tv)"
  # Extraer variables únicas
  try:
      variables = sorted(list(sp.sympify(formula_str).free_symbols), key=lambda x: str(x))
  except Exception as e:
      st.error(f"Error al interpretar la fórmula: {e}")
  # Entrada de valores e incertidumbres
  valores = {}
  incertidumbres = {}
  i = 0
  for var in variables:
    valor = Val_ini[i]
    incertidumbre = u_v_ini[i]
    valores[str(var)] = valor
    incertidumbres[str(var)] = incertidumbre
    i = i + 1
  # Cálculo
  try:
    # Definir símbolos
    simbolos = {str(v): sp.Symbol(str(v)) for v in variables}
    formula_sym = sp.sympify(formula_str)
    # Calcular valor central
    y_val = float(formula_sym.evalf(subs=valores))
    # Derivadas parciales (sensibilidades)
    u_y_squared = 0
    contribuciones = []
    i = 0
    for v in variables:
        var_name = str(v)
        derivada = formula_sym.diff(v)

        sensibilidad = float(derivada.evalf(subs=valores))
        u_i = incertidumbres[var_name]
        contrib = (sensibilidad * u_i)**2
        u_rel_i = u_i / valores[var_name] if valores[var_name] != 0 else np.nan

        contribuciones.append({
            "Variable": var_name,
            "Sensibilidad ∂y/∂x": sensibilidad,
            "Incertidumbre": u_i,
            "Incertidumbre relativa": u_rel_i,
            "Contribución a u(y)²": contrib,
        })
        u_y_squared += contrib
        i = i + 1
    if isinstance(u_y_squared, np.ndarray):
      u_y_squared = u_y_squared.item()
    if isinstance(u_y_squared, list):
      u_y_squared = u_y_squared[0]
    u_y = np.sqrt(u_y_squared) # incertidumbre combinada
    u_rel_y = u_y / y_val if y_val != 0 else np.nan

    # Calcular porcentaje de contribución
    for c in contribuciones:
        c["% Contribución"] = 100 * c["Contribución a u(y)²"] / u_y_squared if u_y_squared > 0 else np.nan
  except Exception as e:
      st.error(f"Ocurrió un error en el cálculo: {e}")
  return u_y, y_val

# -------------------------- Calculo de derivadas -----------------------------#

def cal_derivadas(Val_ini_con):
  (Aesp, Aesp_1, Aesp_2, Aesp_c_Au, Er, Er_1, Er_2, Er_c_Au, Q0, Q0_1, Q0_2,
   Q0_c_Au, alfa, e, e_1, e_2, e_c_Au, k0, k0_1, k0_2, k0_c_Au) = Val_ini_con

  d_Aesp = e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1))))

  d_Aesp_1 = Aesp*e_c_Au*k0_c_Au*(-((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) - (-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) + Aesp*e_c_Au*k0_c_Au*(((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) + (-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2)

  d_Aesp_2 = Aesp*e_c_Au*k0_c_Au*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) - Aesp_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2) + Aesp*e_c_Au*k0_c_Au*(Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) + Aesp_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1))))

  d_Aesp_c_Au = -Aesp*e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au**2*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1))))

  d_Er = Aesp*alfa*e_c_Au*k0_c_Au*(Q0 - 0.429)*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*Er*Er**alfa*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2)

  d_Er_1 = -Aesp*alfa*e_1*e_c_Au*k0_1*k0_c_Au*(Q0_1 - 0.429)/(Aesp_c_Au*Er_1*Er_1**alfa*e*e_2*k0*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) + Aesp*alfa*e_1*e_c_Au*k0_1*k0_c_Au*(Q0_1 - 0.429)*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*Er_1*Er_1**alfa*e*e_2*k0*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2)

  d_Er_2 = Aesp*Aesp_1*alfa*e_c_Au*k0_c_Au*(Q0_2 - 0.429)/(Aesp_2*Aesp_c_Au*Er_2*Er_2**alfa*e*k0*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) - Aesp*Aesp_1*alfa*e_c_Au*k0_c_Au*(Q0_2 - 0.429)*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_2*Aesp_c_Au*Er_2*Er_2**alfa*e*k0*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2)

  d_Er_c_Au = -Aesp*alfa*e_c_Au*k0_c_Au*(Q0_c_Au - 0.429)/(Aesp_c_Au*Er_c_Au*Er_c_Au**alfa*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1))))

  d_Q0 = -Aesp*e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*Er**alfa*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2)

  d_Q0_1 = Aesp*e_1*e_c_Au*k0_1*k0_c_Au/(Aesp_c_Au*Er_1**alfa*e*e_2*k0*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) - Aesp*e_1*e_c_Au*k0_1*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*Er_1**alfa*e*e_2*k0*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2)

  d_Q0_2 = -Aesp*Aesp_1*e_c_Au*k0_c_Au/(Aesp_2*Aesp_c_Au*Er_2**alfa*e*k0*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) + Aesp*Aesp_1*e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_2*Aesp_c_Au*Er_2**alfa*e*k0*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2)

  d_Q0_c_Au = Aesp*e_c_Au*k0_c_Au/(Aesp_c_Au*Er_c_Au**alfa*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1))))

  d_alfa =Aesp*e_c_Au*k0_c_Au*((-Aesp_1*(-(Q0_2 - 0.429)*log(Er_2)/Er_2**alfa + 0.256472073324161/(0.55**alfa*(2*alfa - 1)) - 0.858/(0.55**alfa*(2*alfa - 1)**2))/Aesp_2 + e_1*k0_1*(-(Q0_1 - 0.429)*log(Er_1)/Er_1**alfa + 0.256472073324161/(0.55**alfa*(2*alfa - 1)) - 0.858/(0.55**alfa*(2*alfa - 1)**2))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) - (Q0_c_Au - 0.429)*log(Er_c_Au)/Er_c_Au**alfa + 0.256472073324161/(0.55**alfa*(2*alfa - 1)) - 0.858/(0.55**alfa*(2*alfa - 1)**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) + Aesp*e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))*(-(-Aesp_1*(-(Q0_2 - 0.429)*log(Er_2)/Er_2**alfa + 0.256472073324161/(0.55**alfa*(2*alfa - 1)) - 0.858/(0.55**alfa*(2*alfa - 1)**2))/Aesp_2 + e_1*k0_1*(-(Q0_1 - 0.429)*log(Er_1)/Er_1**alfa + 0.256472073324161/(0.55**alfa*(2*alfa - 1)) - 0.858/(0.55**alfa*(2*alfa - 1)**2))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)*log(Er)/Er**alfa - 0.256472073324161/(0.55**alfa*(2*alfa - 1)) + 0.858/(0.55**alfa*(2*alfa - 1)**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2)

  d_e = -Aesp*e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e**2*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1))))

  d_e_1 = Aesp*e_c_Au*k0_c_Au*(-k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) - k0_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2) + Aesp*e_c_Au*k0_c_Au*(k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) + k0_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1))))

  d_e_2 = Aesp*e_c_Au*k0_c_Au*(-e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2**2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) - e_1*k0_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2**2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) + Aesp*e_c_Au*k0_c_Au*(e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2**2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) + e_1*k0_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2**2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2)

  d_e_c_Au = Aesp*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1))))

  d_k0 = -Aesp*e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0**2*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1))))

  d_k0_1 = Aesp*e_c_Au*k0_c_Au*(-e_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) - e_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2) + Aesp*e_c_Au*k0_c_Au*(e_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) + e_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1))))

  d_k0_2 = Aesp*e_c_Au*k0_c_Au*(-e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) - e_1*k0_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2*k0_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) + Aesp*e_c_Au*k0_c_Au*(e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) + e_1*k0_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2*k0_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2)

  d_k0_c_Au = Aesp*e_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1))))
  (e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))),
  Aesp*e_c_Au*k0_c_Au*(-((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) - (-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) + Aesp*e_c_Au*k0_c_Au*(((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) + (-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2),
  Aesp*e_c_Au*k0_c_Au*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) - Aesp_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2) + Aesp*e_c_Au*k0_c_Au*(Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) + Aesp_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))),
  -Aesp*e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au**2*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))),
  Aesp*alfa*e_c_Au*k0_c_Au*(Q0 - 0.429)*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*Er*Er**alfa*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2),
  -Aesp*alfa*e_1*e_c_Au*k0_1*k0_c_Au*(Q0_1 - 0.429)/(Aesp_c_Au*Er_1*Er_1**alfa*e*e_2*k0*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) + Aesp*alfa*e_1*e_c_Au*k0_1*k0_c_Au*(Q0_1 - 0.429)*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*Er_1*Er_1**alfa*e*e_2*k0*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2),
  Aesp*Aesp_1*alfa*e_c_Au*k0_c_Au*(Q0_2 - 0.429)/(Aesp_2*Aesp_c_Au*Er_2*Er_2**alfa*e*k0*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) - Aesp*Aesp_1*alfa*e_c_Au*k0_c_Au*(Q0_2 - 0.429)*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_2*Aesp_c_Au*Er_2*Er_2**alfa*e*k0*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2),
  -Aesp*alfa*e_c_Au*k0_c_Au*(Q0_c_Au - 0.429)/(Aesp_c_Au*Er_c_Au*Er_c_Au**alfa*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))),
  -Aesp*e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*Er**alfa*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2),
  Aesp*e_1*e_c_Au*k0_1*k0_c_Au/(Aesp_c_Au*Er_1**alfa*e*e_2*k0*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) - Aesp*e_1*e_c_Au*k0_1*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*Er_1**alfa*e*e_2*k0*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2),
  -Aesp*Aesp_1*e_c_Au*k0_c_Au/(Aesp_2*Aesp_c_Au*Er_2**alfa*e*k0*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) + Aesp*Aesp_1*e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_2*Aesp_c_Au*Er_2**alfa*e*k0*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2),
  Aesp*e_c_Au*k0_c_Au/(Aesp_c_Au*Er_c_Au**alfa*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))),
  Aesp*e_c_Au*k0_c_Au*((-Aesp_1*(-(Q0_2 - 0.429)*log(Er_2)/Er_2**alfa + 0.256472073324161/(0.55**alfa*(2*alfa - 1)) - 0.858/(0.55**alfa*(2*alfa - 1)**2))/Aesp_2 + e_1*k0_1*(-(Q0_1 - 0.429)*log(Er_1)/Er_1**alfa + 0.256472073324161/(0.55**alfa*(2*alfa - 1)) - 0.858/(0.55**alfa*(2*alfa - 1)**2))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) - (Q0_c_Au - 0.429)*log(Er_c_Au)/Er_c_Au**alfa + 0.256472073324161/(0.55**alfa*(2*alfa - 1)) - 0.858/(0.55**alfa*(2*alfa - 1)**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) + Aesp*e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))*(-(-Aesp_1*(-(Q0_2 - 0.429)*log(Er_2)/Er_2**alfa + 0.256472073324161/(0.55**alfa*(2*alfa - 1)) - 0.858/(0.55**alfa*(2*alfa - 1)**2))/Aesp_2 + e_1*k0_1*(-(Q0_1 - 0.429)*log(Er_1)/Er_1**alfa + 0.256472073324161/(0.55**alfa*(2*alfa - 1)) - 0.858/(0.55**alfa*(2*alfa - 1)**2))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)*log(Er)/Er**alfa - 0.256472073324161/(0.55**alfa*(2*alfa - 1)) + 0.858/(0.55**alfa*(2*alfa - 1)**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2),
  -Aesp*e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e**2*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))),
  Aesp*e_c_Au*k0_c_Au*(-k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) - k0_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2) + Aesp*e_c_Au*k0_c_Au*(k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) + k0_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))),
  Aesp*e_c_Au*k0_c_Au*(-e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2**2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) - e_1*k0_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2**2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) + Aesp*e_c_Au*k0_c_Au*(e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2**2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) + e_1*k0_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2**2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2),
  Aesp*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))),
  -Aesp*e_c_Au*k0_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0**2*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))),
  Aesp*e_c_Au*k0_c_Au*(-e_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) - e_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2) + Aesp*e_c_Au*k0_c_Au*(e_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) + e_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2*k0_2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))),
  Aesp*e_c_Au*k0_c_Au*(-e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) - e_1*k0_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2*k0_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))) + Aesp*e_c_Au*k0_c_Au*(e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))) + e_1*k0_1*(-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(e_2*k0_2**2*(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2))**2))*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))**2),
  Aesp*e_c_Au*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0_c_Au - 0.429)/Er_c_Au**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(Aesp_c_Au*e*k0*((-Aesp_1*((Q0_2 - 0.429)/Er_2**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/Aesp_2 + e_1*k0_1*((Q0_1 - 0.429)/Er_1**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))/(e_2*k0_2))/(Aesp_1/Aesp_2 - e_1*k0_1/(e_2*k0_2)) + (Q0 - 0.429)/Er**alfa + 0.429/(0.55**alfa*(2*alfa - 1)))))

  derivadas = (d_Aesp, d_Aesp_1, d_Aesp_2, d_Aesp_c_Au, d_Er, d_Er_1, d_Er_2, d_Er_c_Au, d_Q0, d_Q0_1, d_Q0_2, d_Q0_c_Au, d_alfa, d_e, d_e_1, d_e_2, d_e_c_Au, d_k0, d_k0_1, d_k0_2, d_k0_c_Au)

  return derivadas

# ---------------- Correción por picos interferentes ---------------#

def corr_Cn(i, df_final):
    # i ubicación
    # df_final: todos los datos
    delta = 1.0
    df_unico = df_final.iloc[i]
    Nucl = df_unico["Identidad_Verificada_Energia"]
    Area = df_unico["Net_Peak_Area"]
    Interf = df_unico["INTERF"]
    E_Interf = df_unico["E_INTERF"]
    FC = df_unico["FC_GAMM"]
    st.success("Interferente: "+Interf)
    st.success(f"E_Interferente: ", E_Interf)

    if (Interf == "N_A"):
      return Area
    df_filtrado = df_final[(df_final["Identidad_Verificada_Energia"] == Interf) & (df_final["EGKEV"].between(E_Interf - delta, E_Interf + delta))]
    st.success(df_filtrado)
    if df_filtrado.empty:
      st.success("No se encontró inteferente ")
      return Area
    st.success(df_filtrado.iloc[0]["Net_Peak_Area"])
    Area = Area - df_filtrado.iloc[0]["Net_Peak_Area"]*FC

    return Area

# ---------------- Redondeo para concentracióne e incertidumbre ---------------#

def redondear_con_incert(x, u, sig_inc):
    #x: valor nominal
    #porc_u: porcentaje de incertidumbre (ej. 3 = 3%)
    #sig_inc: cifras significativas para la incertidumbre (1 o 2)
    
    u_red = float(f"{u:.{sig_inc}g}")
   
    if u_red <= 0 or np.isnan(u_red):
      st.error(f"La incertidumbre no es válida (u_red = {u_red}).")
      x_red = u_red
    else:
      orden = int(np.floor(np.log10(abs(u_red))))
      #orden = int(np.floor(np.log10(u_red)))
      decimales = -orden
      x_red = round(x, decimales)

    return x_red, u_red
