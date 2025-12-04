# app.py
import streamlit as st
import pandas as pd
import numpy as np
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
from PrecArch import *
from calAAN import *

# Establecer configuración de página
st.set_page_config(
    page_title="Sistema de Análisis k0 - AAN",
    page_icon="🔬",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #93C5FD;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🔬 Sistema de Análisis k0 - AAN</h1>', unsafe_allow_html=True)

# Barra lateral para navegación
st.sidebar.title("🌐 Navegación")
page = st.sidebar.radio(
    "Seleccionar sección:",
    ["📁 Carga de Datos", "⚙️ Configuración", "📊 Procesamiento", "📈 Resultados", "📄 Reporte PDF"]
)


# ============================================
# SECCIÓN 1: CARGA DE DATOS
# ============================================
if page == "📁 Carga de Datos":
    st.markdown('<h2 class="section-header">📁 Carga de Archivos</h2>', unsafe_allow_html=True)
    
    # Crear columnas para la carga de archivos
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("📄 Archivo .RPT de Muestra")
        rpt_file = st.file_uploader("Subir archivo .RPT", type=['rpt', 'RPT'], key="rpt_sample")
        if rpt_file:
            st.success(f"✅ {rpt_file.name} cargado")
            # Leer y mostrar vista previa
            content = rpt_file.getvalue().decode('latin-1')
            st.text_area("Vista previa (primeras 20 líneas):", value='\n'.join(content.split('\n')[:20]), height=200)
    
    with col2:
        st.subheader("📄 Archivo .k0s de Muestra")
        k0s_file = st.file_uploader("Subir archivo .k0s", type=['k0s', 'K0S'], key="k0s_sample")
        if k0s_file:
            st.success(f"✅ {k0s_file.name} cargado")
    
    with col3:
        st.subheader("📄 Archivo .RPT de Au (Comparador)")
        rpt_au_file = st.file_uploader("Subir archivo .RPT de Au", type=['RPT', 'RPT'], key="rpt_au")
        if rpt_au_file:
            st.success(f"✅ {rpt_au_file.name} cargado")

    with col4:
        st.subheader("📄 Archivo .k0s de Au (Comparador)")
        k0s_au_file = st.file_uploader("Subir archivo .k0s de Au", type=['k0s', 'K0S'], key="k0s_au")
        if k0s_au_file:
            st.success(f"✅ {k0s_au_file.name} cargado")

    
    col21, col22 = st.columns(2)

    with col21:
        # Base de datos de Nucléidos
        st.subheader("🗃️ Base de datos de nucléidos")
        db_file = st.file_uploader("Subir Base de Datos (.xlsx)", type=['xlsx'], key="database")
        if db_file:
            st.success(f"✅ Base de datos cargada")

    with col22:
        # Librería de Nucléidos
        st.subheader("📚 Librería de Nucléidos")
        ref_type = st.radio("Seleccionar tipo de nucléidos:", ["Corta (C)", "Media (M)", "Larga (L)"])
        ref_files = st.file_uploader(f"Subir archivo RDN_{ref_type[0]}.xlsx", type=['xlsx'], key="reference")
        if ref_files:
            st.success(f"✅ Archivo cargado")

# ============================================
# SECCIÓN 2: CONFIGURACIÓN
# ============================================
elif page == "⚙️ Configuración":
    st.markdown('<h2 class="section-header">⚙️ Configuración del Análisis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚖️ Parámetros de Masa")
        masa_muestra = st.number_input("Masa de la muestra (g):", min_value=0.0, value=0.2817, step=0.0001, format="%.4f")
        masa_comparador = st.number_input("Masa del comparador Au (μg):", min_value=0.0, value=16.82, step=0.01, format="%.2f")
        
        st.subheader("📐 Geometría")
        geometria = st.radio("Geometría de detección:", ["50 mm", "185 mm"])
        geometria_val = "50" if geometria == "50 mm" else "185"
        
        st.subheader("⏰ Tolerancia de Energía")
        tolerancia = st.slider("Tolerancia de energía (keV):", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    
    with col2:
        st.subheader("🕐 Tiempos de Irradiación")
        col_fecha1, col_hora1 = st.columns(2)
        with col_fecha1:
            fecha_ini = st.date_input("Fecha inicio irradiación:", value=datetime(2025, 9, 26))
        with col_hora1:
            hora_ini = st.time_input("Hora inicio irradiación:", value=datetime.strptime("08:45:00", "%H:%M:%S").time())
        
        col_fecha2, col_hora2 = st.columns(2)
        with col_fecha2:
            fecha_fin = st.date_input("Fecha fin irradiación:", value=datetime(2025, 9, 26))
        with col_hora2:
            hora_fin = st.time_input("Hora fin irradiación:", value=datetime.strptime("09:45:00", "%H:%M:%S").time())
        
        st.subheader("📊 Parámetros de Incertidumbre")
        u_k0 = st.number_input("Incertidumbre k0 (%):", min_value=0.0, max_value=10.0, value=2.8, step=0.1)
        u_e = st.number_input("Incertidumbre eficiencia (%):", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        u_w = st.number_input("Incertidumbre masa (%):", min_value=0.0, max_value=5.0, value=0.01, step=0.01)
    
    # Comparadores para cálculo de alfa
    st.subheader("🔬 Comparadores para Cálculo de f y α")
    col_comp1, col_comp2, col_comp3 = st.columns(3)
    with col_comp1:
        comp1 = st.selectbox("Comparador 1:", ["Au", "Co", "Mo"], index=0)
    with col_comp2:
        comp2 = st.selectbox("Comparador 2:", ["Au", "Co", "Mo"], index=1)
    with col_comp3:
        comp3 = st.selectbox("Comparador 3:", ["Au", "Co", "Mo"], index=2)
    
    st.info("ℹ️ Los comparadores Au, Co y Mo se utilizarán para calcular los parámetros f y α")

# ============================================
# SECCIÓN 3: PROCESAMIENTO
# ============================================
elif page == "📊 Procesamiento":
    st.markdown('<h2 class="section-header">📊 Procesamiento de Datos</h2>', unsafe_allow_html=True)
    
    if st.button("🚀 Iniciar Procesamiento", type="primary", use_container_width=True):
        with st.spinner("Procesando datos..."):
            # Aquí iría la lógica de procesamiento
            # Por ahora mostramos un ejemplo simulado
            
            # Simulación de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps = [
                "Leyendo archivo .k0s de muestra...",
                "Procesando archivo .RPT...",
                "Validando nucleidos...",
                "Calculando tiempos...",
                "Calculando concentraciones...",
                "Calculando incertidumbres...",
                "Generando resultados..."
            ]
            
            for i, step in enumerate(steps):
                progress_bar.progress((i + 1) / len(steps))
                status_text.text(f"📋 {step}")
            
            # Datos de ejemplo
            datos_ejemplo = {
                'Nucleido': ['CE-141', 'SE-75', 'HG-203', 'PA-233', 'CR-51'],
                'Energía (keV)': [145.44, 264.70, 279.19, 312.01, 320.08],
                'Área Neto': [81892, 803, 1844, 79166, 41293],
                'Concentración (ppm)': [26.0, 0.49, 0.30, 4.6, 27.0],
                'Incertidumbre (ppm)': [1.09, 0.08, 0.03, 0.20, 1.15],
                '% Incertidumbre': [4.19, 16.63, 9.22, 4.30, 4.25]
            }
            
            df_ejemplo = pd.DataFrame(datos_ejemplo)
            
            st.success("✅ Procesamiento completado!")
            status_text.text("✅ Procesamiento finalizado")
            
            # Mostrar resultados
            st.subheader("📋 Resultados del Procesamiento")
            st.dataframe(df_ejemplo, use_container_width=True)
            
            # Guardar sesión
            st.session_state['resultados'] = df_ejemplo
            st.session_state['procesado'] = True

# ============================================
# SECCIÓN 4: RESULTADOS
# ============================================
elif page == "📈 Resultados":
    st.markdown('<h2 class="section-header">📈 Visualización de Resultados</h2>', unsafe_allow_html=True)
    
    if 'resultados' in st.session_state:
        df_resultados = st.session_state['resultados']
        
        # Mostrar tabla de resultados
        st.subheader("📊 Tabla de Resultados")
        st.dataframe(df_resultados, use_container_width=True)
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Concentraciones por Elemento")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            bars = ax1.bar(df_resultados['Nucleido'], df_resultados['Concentración (ppm)'])
            ax1.set_ylabel('Concentración (ppm)')
            ax1.set_xlabel('Nucleido')
            ax1.set_title('Concentraciones Calculadas')
            ax1.tick_params(axis='x', rotation=45)
            
            # Añadir etiquetas de valor
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig1)
        
        with col2:
            st.subheader("📊 Incertidumbre Relativa")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            colors = ['#FF6B6B' if x > 10 else '#4ECDC4' for x in df_resultados['% Incertidumbre']]
            bars = ax2.bar(df_resultados['Nucleido'], df_resultados['% Incertidumbre'], color=colors)
            ax2.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Límite 10%')
            ax2.set_ylabel('Incertidumbre Relativa (%)')
            ax2.set_xlabel('Nucleido')
            ax2.set_title('Incertidumbre por Elemento')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()
            
            st.pyplot(fig2)
        
        # Estadísticas resumidas
        st.subheader("📋 Resumen Estadístico")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Número de Elementos", len(df_resultados))
        with col_stat2:
            avg_conc = df_resultados['Concentración (ppm)'].mean()
            st.metric("Concentración Promedio", f"{avg_conc:.2f} ppm")
        with col_stat3:
            avg_uncert = df_resultados['% Incertidumbre'].mean()
            st.metric("Incertidumbre Promedio", f"{avg_uncert:.2f}%")
        with col_stat4:
            max_conc = df_resultados['Concentración (ppm)'].max()
            st.metric("Concentración Máxima", f"{max_conc:.2f} ppm")
        
        # Botón para exportar
        st.download_button(
            label="📥 Descargar Resultados (Excel)",
            data=df_resultados.to_csv(index=False).encode('utf-8'),
            file_name="resultados_k0_analisis.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("⚠️ No hay resultados disponibles. Por favor, ejecute el procesamiento primero.")

# ============================================
# SECCIÓN 5: REPORTE PDF
# ============================================
elif page == "📄 Reporte PDF":
    st.markdown('<h2 class="section-header">📄 Generación de Reporte</h2>', unsafe_allow_html=True)
    
    # Información del reporte
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        proyecto = st.text_input("Nombre del Proyecto:", value="Evaluación Elemental por k0-INAA")
        operador = st.text_input("Nombre del Operador:", value="José Oliden")
        laboratorio = st.text_input("Laboratorio:", value="Laboratorio de Análisis por Activación Neutrónica")
    
    with col_info2:
        muestra_id = st.text_input("ID de Muestra:", value="6824a2131025G50")
        fecha_analisis = st.date_input("Fecha de Análisis:", value=datetime.now())
        metodo = st.selectbox("Método:", ["k0-INAA", "k0-EDXRF", "k0-PIXE"])
    
    # Parámetros del reporte
    st.subheader("⚙️ Configuración del Reporte")
    incluir_graficos = st.checkbox("Incluir gráficos", value=True)
    incluir_datos_crudos = st.checkbox("Incluir datos crudos", value=False)
    formato = st.radio("Formato del reporte:", ["PDF", "HTML", "Word"], horizontal=True)
    
    # Vista previa
    st.subheader("👁️ Vista Previa del Reporte")
    if st.button("🔄 Generar Vista Previa", type="secondary"):
        with st.expander("📋 Contenido del Reporte", expanded=True):
            st.markdown(f"""
            ## Reporte de Análisis k0
            
            ### Información General
            - **Proyecto:** {proyecto}
            - **Operador:** {operador}
            - **Laboratorio:** {laboratorio}
            - **ID Muestra:** {muestra_id}
            - **Fecha de Análisis:** {fecha_analisis.strftime('%d/%m/%Y')}
            - **Método:** {metodo}
            
            ### Parámetros de Análisis
            - **Geometría:** 50 mm
            - **Comparadores:** Au, Co, Mo
            - **Fecha Irradiación:** 26/09/2025 08:45:00 - 26/09/2025 09:45:00
            - **Masa muestra:** 0.2817 g
            - **Masa comparador Au:** 16.82 μg
            
            ### Resumen de Resultados
            - **Número de elementos detectados:** 17
            - **Concentración promedio:** 514.2 ppm
            - **Incertidumbre promedio:** 6.3%
            
            ### Próximos pasos
            1. Verificar resultados
            2. Validar con estándares
            3. Archivar reporte
            """)
    
    # Botón para generar reporte completo
    if st.button("🖨️ Generar Reporte Completo", type="primary", use_container_width=True):
        st.success("✅ Reporte generado exitosamente!")
        st.info("📄 El reporte se ha generado y está listo para descargar")
        
        # Crear un archivo de ejemplo (en realidad sería un PDF generado)
        reporte_texto = f"""
        REPORTE DE ANÁLISIS k0-INAA
        ============================
        
        Proyecto: {proyecto}
        Operador: {operador}
        Laboratorio: {laboratorio}
        Muestra ID: {muestra_id}
        Fecha: {fecha_analisis.strftime('%d/%m/%Y')}
        
        RESULTADOS:
        -----------
        
        Este es un reporte de ejemplo generado por el sistema.
        
        Para generar el reporte PDF completo, se necesitaría implementar
        la biblioteca ReportLab o similar.
        """
        
        st.download_button(
            label="📥 Descargar Reporte (.txt)",
            data=reporte_texto.encode('utf-8'),
            file_name=f"reporte_{muestra_id}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Pie de página
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280;'>
        <p>Sistema de Análisis k0 - AAN v7.0 | Desarrollado para análisis por activación neutrónica</p>
        <p>© 2024 Laboratorio de Análisis por Activación Neutrónica</p>
    </div>
    """,
    unsafe_allow_html=True
)
