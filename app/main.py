# app/main.py

import os, sys

# parent directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from github import Github
import pandas as pd
from app.utils import load_data

st.set_page_config(page_title="Análisis Motor Equino", layout="wide")

st.title("Plataforma de Análisis Motor Equino")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Datos del Dr. Veterinario")
    df_vet = load_data("datasets/example_vet.csv")
    st.dataframe(df_vet)

with col2:
    st.subheader("Datos del Modelo AI")
    df_model = load_data("datasets/example_modelai.csv")
    st.dataframe(df_model)        

st.markdown("---")
st.subheader("Añadir Registro Veterinario")

with st.form("form_vet"):
    caballo_id = st.text_input("Caballo_ID")
    nombre     = st.text_input("Nombre")
    raza       = st.text_input("Raza")
    sexo       = st.selectbox("Sexo", ["Macho","Hembra"])
    edad       = st.number_input("Edad", min_value=0, step=1)
    lrmd       = st.selectbox("LRMD", [0,1,2,3])
    lrmi       = st.selectbox("LRMI", [0,1,2,3])
    cmd        = st.selectbox("CMD",  [0,1,2,3])
    cmi        = st.selectbox("CMI",  [0,1,2,3])
    pflrmd     = st.selectbox("PFLRMD",[0,1,2,3])
    pflrmi     = st.selectbox("PFLRMI",[0,1,2,3])
    pfcmd      = st.selectbox("PFCMD",[0,1,2,3])
    pfcmi      = st.selectbox("PFCMI",[0,1,2,3])
    submit_vet = st.form_submit_button("Guardar en base de datos")

# data for vet results
if submit_vet:
    # store in csv
    df = load_data("datasets/example_vet.csv")
    new_row = {
        "Caballo_ID": caballo_id, "Nombre": nombre, "Raza": raza,
        "Sexo": sexo, "Edad": edad, "Comentarios_generales": "",
        "LRMD": lrmd, "LRMI": lrmi, "CMD": cmd, "CMI": cmi,
        "PFLRMD": pflrmd, "PFLRMI": pflrmi, "PFCMD": pfcmd, "PFCMI": pfcmi
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    csv_text = df.to_csv(index=False)

    # commit in gh
    gh_repo = Github(st.secrets["github"]["token"]) \
              .get_repo("faviles11/equine-motor-analysis")
    file = gh_repo.get_contents("datasets/example_vet.csv", ref="main")
    gh_repo.update_file(
        path=file.path,
        message=f"chore: añade registro VET {caballo_id}",
        content=csv_text,
        sha=file.sha,
        branch="main"
    )
    st.success(f"✅ Registro Veterinario {caballo_id} agregado en base de datos")


# data for ai model results
st.markdown("---")
st.subheader("Añadir Registro Modelo AI")

with st.form("form_model"):
    # mismos campos de metadatos + resultados
    caballo_id_m = st.text_input("Caballo_ID", key="m_id")
    nombre_m     = st.text_input("Nombre", key="m_nombre")
    raza_m       = st.text_input("Raza", key="m_raza")
    sexo_m       = st.selectbox("Sexo", ["Macho","Hembra"], key="m_sexo")
    edad_m       = st.number_input("Edad", min_value=0, step=1, key="m_edad")
    lrmd_m       = st.selectbox("LRMD", [0,1,2,3], key="m_lrmd")
    lrmi_m       = st.selectbox("LRMI", [0,1,2,3], key="m_lrmi")
    cmd_m        = st.selectbox("CMD",  [0,1,2,3], key="m_cmd")
    cmi_m        = st.selectbox("CMI",  [0,1,2,3], key="m_cmi")
    pflrmd_m     = st.selectbox("PFLRMD",[0,1,2,3], key="m_pflrmd")
    pflrmi_m     = st.selectbox("PFLRMI",[0,1,2,3], key="m_pflrmi")
    pfcmd_m      = st.selectbox("PFCMD",[0,1,2,3], key="m_pfcmd")
    pfcmi_m      = st.selectbox("PFCMI",[0,1,2,3], key="m_pfcmi")
    submit_mod   = st.form_submit_button("Guardar en base de datos")

if submit_mod:
    # store in csv
    df = load_data("datasets/example_modelai.csv")
    new_row = {
        "Caballo_ID": caballo_id_m, "Nombre": nombre_m, "Raza": raza_m,
        "Sexo": sexo_m, "Edad": edad_m, "Comentarios_generales": "",
        "LRMD": lrmd_m, "LRMI": lrmi_m, "CMD": cmd_m, "CMI": cmi_m,
        "PFLRMD": pflrmd_m, "PFLRMI": pflrmi_m, "PFCMD": pfcmd_m, "PFCMI": pfcmi_m
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    csv_text = df.to_csv(index=False)

    # commit in gh
    gh_repo = Github(st.secrets["github"]["token"]) \
              .get_repo("faviles11/equine-motor-analysis")
    file = gh_repo.get_contents("datasets/example_modelai.csv", ref="main")
    gh_repo.update_file(
        path=file.path,
        message=f"chore: añade registro AI {caballo_id_m}",
        content=csv_text,
        sha=file.sha,
        branch="main"
    )
    st.success(f"✅ Registro AI {caballo_id_m} agregado en base de datos")    