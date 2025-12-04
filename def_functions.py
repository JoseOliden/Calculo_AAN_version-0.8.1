# Función para limpiar nombres
def limpiar_nombre(texto):
    if pd.isna(texto):
        return ""
    return str(texto).upper().replace('-', '').replace(' ', '').strip()

# Funciones de cálculo (simplificadas para Streamlit)
def Aesp(Cn_i, w_i, lam, tr, td, ti, tv, e):
    C_i = lam / (1 - np.exp(-lam * tr))
    D_i = np.exp(lam * td)
    H_i = tr / tv
    S_i = 1 - np.exp(-lam * ti)
    return Cn_i * D_i * C_i * H_i / (S_i * w_i)

def equations(vars, *par):
    alfa = vars[0]
    Aesp_1, k0_1, e_1, Er_1, Q0_1, Aesp_2, k0_2, e_2, Er_2, Q0_2, Aesp_3, k0_3, e_3, Er_3, Q0_3 = par
    eq1 = ((1 - (Aesp_2 / Aesp_1) * (k0_1 / k0_2) * (e_1 / e_2)) ** (-1) - 
           (1 - (Aesp_3 / Aesp_1) * (k0_1 / k0_3) * (e_1 / e_3)) ** (-1)) * (Q0_1 - 0.429) / (Er_1 ** alfa) - \
          ((1 - (Aesp_2 / Aesp_1) * (k0_1 / k0_2) * (e_1 / e_2)) ** (-1)) * (Q0_2 - 0.429) / (Er_2 ** alfa) + \
          ((1 - (Aesp_3 / Aesp_1) * (k0_1 / k0_3) * (e_1 / e_3)) ** (-1)) * (Q0_3 - 0.429) / (Er_3 ** alfa)
    return [eq1]
