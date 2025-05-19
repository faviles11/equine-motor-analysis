# app/main.py

import os, sys

# parent directory to python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

import streamlit as st
from github import Github
import pandas as pd
from app.utils import load_data
from sklearn.metrics import cohen_kappa_score

st.set_page_config(page_title="Análisis Motor Equino", layout="wide")

st.title("Plataforma de Análisis Motor Equino")

# show data in table
st.subheader("Datos del Dr. Veterinario")
df_vet = load_data("datasets/example_vet.csv")
st.dataframe(df_vet)

st.subheader("Datos del Modelo AI")
df_model = load_data("datasets/example_modelai.csv")
st.dataframe(df_model)

st.markdown("---")

# buttons for forms
if "show_vet"   not in st.session_state: st.session_state.show_vet   = False
if "show_mod"   not in st.session_state: st.session_state.show_mod   = False

col1_btn, col2_btn = st.columns(2)
with col1_btn:
    if st.button("➕ Agregar registro Veterinario"):
        st.session_state.show_vet = not st.session_state.show_vet
with col2_btn:
    if st.button("➕ Agregar registro Modelo AI"):
        st.session_state.show_mod = not st.session_state.show_mod

# vet form
if st.session_state.show_vet:
    st.markdown("## Formulario: Veterinario")
    with st.form("form_vet"):
        nombre     = st.text_input("Nombre")
        raza       = st.text_input("Raza")
        sexo       = st.selectbox("Sexo", ["Macho","Hembra"])
        edad       = st.number_input("Edad", min_value=0, step=1)
        analisis   = st.text_input("Análisis clínico", key="m_analisis")

        st.markdown("**Asimetría vertical de Cabeza**")
        cabeza_lrmd   = st.selectbox("Cabeza_LRMD",   [0,1,2,3])
        cabeza_lrmi   = st.selectbox("Cabeza_LRMI",   [0,1,2,3])
        cabeza_cmd    = st.selectbox("Cabeza_CMD",    [0,1,2,3])
        cabeza_cmi    = st.selectbox("Cabeza_CMI",    [0,1,2,3])
        cabeza_pflrmd = st.selectbox("Cabeza_PFLRMD", [0,1,2,3])
        cabeza_pflrmi = st.selectbox("Cabeza_PFLRMI", [0,1,2,3])
        cabeza_pfcmd  = st.selectbox("Cabeza_PFCMD",  [0,1,2,3])
        cabeza_pfcmi  = st.selectbox("Cabeza_PFCMI",  [0,1,2,3])

        st.markdown("**Asimetría vertical de Pelvis**")
        pelvis_lrmd   = st.selectbox("Pelvis_LRMD",   [0,1,2,3])
        pelvis_lrmi   = st.selectbox("Pelvis_LRMI",   [0,1,2,3])
        pelvis_cmd    = st.selectbox("Pelvis_CMD",    [0,1,2,3])
        pelvis_cmi    = st.selectbox("Pelvis_CMI",    [0,1,2,3])
        pelvis_pflrmd = st.selectbox("Pelvis_PFLRMD", [0,1,2,3])
        pelvis_pflrmi = st.selectbox("Pelvis_PFLRMI", [0,1,2,3])
        pelvis_pfcmd  = st.selectbox("Pelvis_PFCMD",  [0,1,2,3])
        pelvis_pfcmi  = st.selectbox("Pelvis_PFCMI",  [0,1,2,3])

        submit_vet = st.form_submit_button("Guardar Vet")

    if submit_vet:
        df = load_data("datasets/example_vet.csv")
        new_id = str(len(df) + 1).zfill(3)
        new_row = {
            "Caballo_ID": new_id, "Nombre": nombre, "Raza": raza,
            "Sexo": sexo, "Edad": edad, "Analisis_clinico": analisis,
            # head
            "Cabeza_LRMD": cabeza_lrmd,   "Cabeza_LRMI": cabeza_lrmi,
            "Cabeza_CMD": cabeza_cmd,     "Cabeza_CMI": cabeza_cmi,
            "Cabeza_PFLRMD": cabeza_pflrmd, "Cabeza_PFLRMI": cabeza_pflrmi,
            "Cabeza_PFCMD": cabeza_pfcmd, "Cabeza_PFCMI": cabeza_pfcmi,
            # pelvis
            "Pelvis_LRMD": pelvis_lrmd,   "Pelvis_LRMI": pelvis_lrmi,
            "Pelvis_CMD": pelvis_cmd,     "Pelvis_CMI": pelvis_cmi,
            "Pelvis_PFLRMD": pelvis_pflrmd, "Pelvis_PFLRMI": pelvis_pflrmi,
            "Pelvis_PFCMD": pelvis_pfcmd, "Pelvis_PFCMI": pelvis_pfcmi,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        csv_text = df.to_csv(index=False)

        gh = Github(st.secrets["github"]["token"])
        repo = gh.get_repo("faviles11/equine-motor-analysis")
        f   = repo.get_contents("datasets/example_vet.csv", ref="main")
        repo.update_file(path=f.path, message=f"chore: añade registro VET {nombre}",
                         content=csv_text, sha=f.sha, branch="main")
        with open(os.path.join(repo_root, "datasets", "example_vet.csv"), "w", encoding="utf-8") as fo:
            fo.write(csv_text)

        st.success("✅ Veterinario guardado")
        st.cache_data.clear()
        st.rerun()

# ai form
if st.session_state.show_mod:
    st.markdown("## Formulario: Modelo AI")
    with st.form("form_mod"):
        nombre_m     = st.text_input("Nombre",    key="m_nombre")
        raza_m       = st.text_input("Raza",      key="m_raza")
        sexo_m       = st.selectbox("Sexo", ["Macho","Hembra"], key="m_sexo")
        edad_m       = st.number_input("Edad", min_value=0, step=1, key="m_edad")
        analisis_m   = st.text_input("Análisis clínico", key="m_analisis")

        st.markdown("**Asimetría vertical de Cabeza**")
        cabeza_lrmd_m   = st.selectbox("Cabeza_LRMD",   [0,1,2,3], key="m_cabeza_lrmd")
        cabeza_lrmi_m   = st.selectbox("Cabeza_LRMI",   [0,1,2,3], key="m_cabeza_lrmi")
        cabeza_cmd_m    = st.selectbox("Cabeza_CMD",    [0,1,2,3], key="m_cabeza_cmd")
        cabeza_cmi_m    = st.selectbox("Cabeza_CMI",    [0,1,2,3], key="m_cabeza_cmi")
        cabeza_pflrmd_m = st.selectbox("Cabeza_PFLRMD", [0,1,2,3], key="m_cabeza_pflrmd")
        cabeza_pflrmi_m = st.selectbox("Cabeza_PFLRMI", [0,1,2,3], key="m_cabeza_pflrmi")
        cabeza_pfcmd_m  = st.selectbox("Cabeza_PFCMD",  [0,1,2,3], key="m_cabeza_pfcmd")
        cabeza_pfcmi_m  = st.selectbox("Cabeza_PFCMI",  [0,1,2,3], key="m_cabeza_pfcmi")

        st.markdown("**Asimetría vertical de Pelvis**")
        pelvis_lrmd_m   = st.selectbox("Pelvis_LRMD",   [0,1,2,3], key="m_pelvis_lrmd")
        pelvis_lrmi_m   = st.selectbox("Pelvis_LRMI",   [0,1,2,3], key="m_pelvis_lrmi")
        pelvis_cmd_m    = st.selectbox("Pelvis_CMD",    [0,1,2,3], key="m_pelvis_cmd")
        pelvis_cmi_m    = st.selectbox("Pelvis_CMI",    [0,1,2,3], key="m_pelvis_cmi")
        pelvis_pflrmd_m = st.selectbox("Pelvis_PFLRMD", [0,1,2,3], key="m_pelvis_pflrmd")
        pelvis_pflrmi_m = st.selectbox("Pelvis_PFLRMI", [0,1,2,3], key="m_pelvis_pflrmi")
        pelvis_pfcmd_m  = st.selectbox("Pelvis_PFCMD",  [0,1,2,3], key="m_pelvis_pfcmd")
        pelvis_pfcmi_m  = st.selectbox("Pelvis_PFCMI",  [0,1,2,3], key="m_pelvis_pfcmi")

        submit_mod = st.form_submit_button("Guardar AI")

    if submit_mod:
        df = load_data("datasets/example_modelai.csv")
        new_id_m = str(len(df) + 1).zfill(3)
        new_row = {
            "Caballo_ID": new_id_m,
            "Nombre": nombre_m,
            "Raza": raza_m,
            "Sexo": sexo_m,
            "Edad": edad_m,
            "Analisis-clinico": analisis_m,
            # head
            "Cabeza_LRMD": cabeza_lrmd_m,
            "Cabeza_LRMI": cabeza_lrmi_m,
            "Cabeza_CMD": cabeza_cmd_m,
            "Cabeza_CMI": cabeza_cmi_m,
            "Cabeza_PFLRMD": cabeza_pflrmd_m,
            "Cabeza_PFLRMI": cabeza_pflrmi_m,
            "Cabeza_PFCMD": cabeza_pfcmd_m,
            "Cabeza_PFCMI": cabeza_pfcmi_m,
            # pelvis
            "Pelvis_LRMD": pelvis_lrmd_m,
            "Pelvis_LRMI": pelvis_lrmi_m,
            "Pelvis_CMD": pelvis_cmd_m,
            "Pelvis_CMI": pelvis_cmi_m,
            "Pelvis_PFLRMD": pelvis_pflrmd_m,
            "Pelvis_PFLRMI": pelvis_pflrmi_m,
            "Pelvis_PFCMD": pelvis_pfcmd_m,
            "Pelvis_PFCMI": pelvis_pfcmi_m,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        csv_text = df.to_csv(index=False)

        gh = Github(st.secrets["github"]["token"])
        repo = gh.get_repo("faviles11/equine-motor-analysis")
        f   = repo.get_contents("datasets/example_modelai.csv", ref="main")
        repo.update_file(path=f.path, message=f"chore: añade registro AI {nombre_m}",
                         content=csv_text, sha=f.sha, branch="main")
        with open(os.path.join(repo_root, "datasets", "example_modelai.csv"), "w", encoding="utf-8") as fo:
            fo.write(csv_text)

        st.success("✅ Modelo AI guardado")
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# cohen's kappa analysis
st.header("Cálculo de Kappa de Cohen")

param_cols = [
    "Cabeza_LRMD","Cabeza_LRMI","Cabeza_CMD","Cabeza_CMI",
    "Cabeza_PFLRMD","Cabeza_PFLRMI","Cabeza_PFCMD","Cabeza_PFCMI",
    "Pelvis_LRMD","Pelvis_LRMI","Pelvis_CMD","Pelvis_CMI",
    "Pelvis_PFLRMD","Pelvis_PFLRMI","Pelvis_PFCMD","Pelvis_PFCMI",
]

# checking data frames caballo id consistency
df_v = df_vet.set_index("Caballo_ID").loc[df_model["Caballo_ID"]]
df_m = df_model.set_index("Caballo_ID").loc[df_v.index]

# calculating cohen's kappa score
kappas = {
    col: cohen_kappa_score(df_v[col], df_m[col])
    for col in param_cols
}

kappa_df = pd.DataFrame.from_dict(kappas, orient="index", columns=["Kappa"])
st.dataframe(kappa_df)

# dashboards
st.header("Dashboard de Acuerdos")

# 3.1 cohen's kappa Bar chart 
st.subheader("Kappa por Parámetro")
st.bar_chart(kappa_df["Kappa"])

# 3.2 discrepancies line chart
st.subheader("Discrepancias Totales")
diffs = (df_v[param_cols] - df_m[param_cols]).abs().stack()
st.line_chart(diffs.value_counts().sort_index())

# 3.3 summary metrics
st.subheader("Resumen Global")
mean_kappa = kappa_df["Kappa"].mean()
st.metric("Kappa Medio", f"{mean_kappa:.2f}")
      