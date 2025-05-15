# app/main.py

import streamlit as st
from app.utils import load_data

st.set_page_config(page_title="Análisis Motor Equino", layout="centered")

st.title("Plataforma de Análisis Motor Equino")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Datos del Dr. Veterinario")
    vet_file = st.file_uploader("Sube example_vet.csv", type="csv", key="vet")
    if vet_file:
        df_vet = load_data(vet_file)
        st.dataframe(df_vet)

with col2:
    st.subheader("Datos del Modelo AI")
    model_file = st.file_uploader("Sube example_modelai.csv", type="csv", key="model")
    if model_file:
        df_model = load_data(model_file)
        st.dataframe(df_model)        
