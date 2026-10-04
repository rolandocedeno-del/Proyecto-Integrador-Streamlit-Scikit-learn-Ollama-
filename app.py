from __future__ import annotations
import streamlit as st
import pandas as pd
from __future__ import annotations
import sys
import os

# Añade la ruta del directorio raíz del proyecto al PATH de Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importaciones posteriores

import streamlit as st
import pandas as pd

from modules.data import load_default_data, load_uploaded_data, dataset_profile
# ... resto de tus imports

from modules.data import load_default_data, load_uploaded_data, dataset_profile
from modules.analytics import (
    missing_report,
    numeric_summary,
    target_correlations,
    train_regression_demo,
    build_agent_context,
)
from modules.visualizations import histogram, scatter, correlation_heatmap, predicted_vs_real
from modules.agent import LocalDataAgent, offline_summary

st.set_page_config(page_title="Demo IA + Datos", page_icon="🤖", layout="wide")

st.title("Demo · Explorador de datos con agente local")
st.caption("Streamlit + Scikit-learn + Ollama · ejemplo básico del Proyecto Final Integrador")

with st.sidebar:
    st.header("Configuración")
    uploaded = st.file_uploader("CSV opcional", type=["csv"])
    model_name = st.text_input("Modelo Ollama", value="llama3.2:3b")

try:
    df = load_uploaded_data(uploaded) if uploaded else load_default_data()
except Exception as exc:
    st.error(f"No fue posible cargar el dataset: {exc}")
    st.stop()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
    st.error("El demo requiere al menos una variable numérica.")
    st.stop()

default_target = "target" if "target" in numeric_cols else numeric_cols[-1]
target = st.sidebar.selectbox("Variable objetivo", numeric_cols, index=numeric_cols.index(default_target))

profile = dataset_profile(df)
model_result = None
try:
    model_result = train_regression_demo(df, target)
except Exception:
    pass

agent = LocalDataAgent(model=model_name)
ollama_ok, installed_models, ollama_detail = agent.status()
with st.sidebar:
    st.markdown("---")
    if ollama_ok:
        st.success("Ollama disponible")
        if installed_models:
            st.caption("Modelos: " + ", ".join(installed_models[:5]))
    else:
        st.warning("Ollama no detectado")
        st.caption("La aplicación continúa funcionando sin el agente generativo.")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", profile["rows"])
c2.metric("Variables", profile["columns"])
c3.metric("Valores nulos", profile["missing"])
c4.metric("Duplicados", profile["duplicates"])

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Datos", "📊 Exploración", "📈 Modelo", "🤖 Agente", "🧩 Arquitectura"
])

with tab1:
    st.subheader("Dataset de trabajo")
    if uploaded is None:
        st.info(
            "Dataset incluido: Ventas Minorista de Scikit-learn, exportado como CSV con variables sin escalado. "
            "Se usa exclusivamente como ejemplo académico de regresión."
        )
    else:
        st.info("Se está utilizando el CSV cargado desde la interfaz.")
    st.dataframe(df.head(25), use_container_width=True, hide_index=True)

    st.markdown("#### Calidad de datos")
    st.dataframe(missing_report(df), use_container_width=True, hide_index=True)

    st.markdown("#### Resumen numérico")
    st.dataframe(numeric_summary(df), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Exploración visual")
    xcol = st.selectbox("Variable para distribución", numeric_cols, index=0)
    st.plotly_chart(histogram(df, xcol), use_container_width=True)

    candidates = [c for c in numeric_cols if c != target]
    if candidates:
        xscatter = st.selectbox("Variable explicativa", candidates, index=0)
        st.plotly_chart(scatter(df, xscatter, target), use_container_width=True)

    st.markdown("#### Correlaciones con la variable objetivo")
    st.dataframe(target_correlations(df, target).head(10), use_container_width=True, hide_index=True)
    st.plotly_chart(correlation_heatmap(df), use_container_width=True)

with tab3:
    st.subheader("Modelo de regresión lineal")
    if model_result is None:
        st.warning("No fue posible entrenar el modelo con la configuración actual.")
    else:
        m = model_result["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{m['r2']:.3f}")
        c2.metric("MAE", f"{m['mae']:.3f}")
        c3.metric("RMSE", f"{m['rmse']:.3f}")
        st.caption(f"Entrenamiento: {model_result['n_train']} registros · Prueba: {model_result['n_test']} registros")
        st.plotly_chart(predicted_vs_real(model_result["predictions"]), use_container_width=True)
        st.markdown("#### Coeficientes del modelo")
        st.dataframe(model_result["coefficients"], use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Agente local para interpretación")
    context = build_agent_context(df, target, model_result)
    st.caption("El agente recibe un contexto estadístico resumido; no recibe el dataset completo.")

    presets = [
        "Resume los principales hallazgos del dataset.",
        "¿Qué variables presentan mayor relación con la variable objetivo?",
        "Interpreta las métricas del modelo de regresión.",
        "¿Qué análisis adicional sería razonable realizar?",
    ]
    choice = st.selectbox("Pregunta sugerida", ["Escribir otra pregunta..."] + presets)
    question = st.text_area(
        "Pregunta",
        value="" if choice == "Escribir otra pregunta..." else choice,
        height=100,
    )

    if st.button("Analizar con el agente", type="primary"):
        if not question.strip():
            st.warning("Escribe una pregunta.")
        elif ollama_ok:
            with st.spinner("Consultando el modelo local..."):
                try:
                    answer = agent.ask(question.strip(), context)
                    st.markdown(answer)
                except Exception as exc:
                    st.error(f"Ollama respondió con un error: {exc}")
                    st.markdown(offline_summary(context))
        else:
            st.markdown(offline_summary(context))
            st.caption("Respuesta de respaldo: Ollama no está disponible en este equipo.")

with tab5:
    st.subheader("Arquitectura modular")
    st.code(
        """CSV / archivo cargado
        ↓
modules/data.py
        ↓
modules/analytics.py
   ↙            ↘
visualizations.py  contexto resumido
   ↓            ↓
Streamlit ← modules/agent.py → Ollama local""",
        language="text",
    )
    st.markdown(
        """
- **Imperativo:** flujo de carga, validaciones y decisiones de la interfaz.
- **Funcional:** funciones de perfilado, métricas, correlaciones y visualización.
- **Orientado a objetos:** clase `LocalDataAgent` para encapsular configuración y comunicación con Ollama.
- **Modularidad:** cada responsabilidad se mantiene en un archivo separado.
        """
    )

st.markdown("---")
st.caption("Demo académico. No constituye una herramienta real para negocios.")
