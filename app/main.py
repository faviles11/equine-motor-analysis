# app/main.py

import numpy as np
import os, sys

# parent directory to python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

import streamlit as st
from github import Github
import pandas as pd
from app.utils import load_data
from sklearn.metrics import cohen_kappa_score
import plotly.express as px
from plotly.colors import sample_colorscale

st.set_page_config(page_title="Análisis Motor Equino", layout="wide")

st.title("Plataforma de Análisis Motor Equino")
st.subheader("Daniel Avilés Chinchilla")
# refresh cache for aditional analysis
if st.button("🔄 Actualizar datos"):
    st.cache_data.clear()
    st.rerun()
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
        condicion  = st.text_input("Condición Corporal", key="m_condicion")

        st.markdown("**Asimetría vertical de Cabeza**")
        cabeza_lrmd   = st.selectbox("Cabeza_LRMD",   [0,1,2,3,4])
        cabeza_lrmi   = st.selectbox("Cabeza_LRMI",   [0,1,2,3,4])
        cabeza_cmd    = st.selectbox("Cabeza_CMD",    [0,1,2,3,4])
        cabeza_cmi    = st.selectbox("Cabeza_CMI",    [0,1,2,3,4])
        cabeza_pflrmd = st.selectbox("Cabeza_PFLRMD", [0,1,2,3,4])
        cabeza_pflrmi = st.selectbox("Cabeza_PFLRMI", [0,1,2,3,4])
        cabeza_pfcmd  = st.selectbox("Cabeza_PFCMD",  [0,1,2,3,4])
        cabeza_pfcmi  = st.selectbox("Cabeza_PFCMI",  [0,1,2,3,4])

        st.markdown("**Asimetría vertical de Pelvis**")
        pelvis_lrmd   = st.selectbox("Pelvis_LRMD",   [0,1,2,3,4])
        pelvis_lrmi   = st.selectbox("Pelvis_LRMI",   [0,1,2,3,4])
        pelvis_cmd    = st.selectbox("Pelvis_CMD",    [0,1,2,3,4])
        pelvis_cmi    = st.selectbox("Pelvis_CMI",    [0,1,2,3,4])
        pelvis_pflrmd = st.selectbox("Pelvis_PFLRMD", [0,1,2,3,4])
        pelvis_pflrmi = st.selectbox("Pelvis_PFLRMI", [0,1,2,3,4])
        pelvis_pfcmd  = st.selectbox("Pelvis_PFCMD",  [0,1,2,3,4])
        pelvis_pfcmi  = st.selectbox("Pelvis_PFCMI",  [0,1,2,3,4])

        submit_vet = st.form_submit_button("Guardar Vet")

    if submit_vet:
        df = load_data("datasets/example_vet.csv")
        new_id = str(len(df) + 1).zfill(3)
        new_row = {
            "Caballo_ID": new_id, "Nombre": nombre, "Raza": raza,
            "Sexo": sexo, "Edad": edad, "Analisis_clinico": analisis,
            "Condicion_Corporal": condicion,
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
        cabeza_lrmd_m   = st.selectbox("Cabeza_LRMD",   [0,1,2,3,4], key="m_cabeza_lrmd")
        cabeza_lrmi_m   = st.selectbox("Cabeza_LRMI",   [0,1,2,3,4], key="m_cabeza_lrmi")
        cabeza_cmd_m    = st.selectbox("Cabeza_CMD",    [0,1,2,3,4], key="m_cabeza_cmd")
        cabeza_cmi_m    = st.selectbox("Cabeza_CMI",    [0,1,2,3,4], key="m_cabeza_cmi")
        cabeza_pflrmd_m = st.selectbox("Cabeza_PFLRMD", [0,1,2,3,4], key="m_cabeza_pflrmd")
        cabeza_pflrmi_m = st.selectbox("Cabeza_PFLRMI", [0,1,2,3,4], key="m_cabeza_pflrmi")
        cabeza_pfcmd_m  = st.selectbox("Cabeza_PFCMD",  [0,1,2,3,4], key="m_cabeza_pfcmd")
        cabeza_pfcmi_m  = st.selectbox("Cabeza_PFCMI",  [0,1,2,3,4], key="m_cabeza_pfcmi")

        st.markdown("**Asimetría vertical de Pelvis**")
        pelvis_lrmd_m   = st.selectbox("Pelvis_LRMD",   [0,1,2,3,4], key="m_pelvis_lrmd")
        pelvis_lrmi_m   = st.selectbox("Pelvis_LRMI",   [0,1,2,3,4], key="m_pelvis_lrmi")
        pelvis_cmd_m    = st.selectbox("Pelvis_CMD",    [0,1,2,3,4], key="m_pelvis_cmd")
        pelvis_cmi_m    = st.selectbox("Pelvis_CMI",    [0,1,2,3,4], key="m_pelvis_cmi")
        pelvis_pflrmd_m = st.selectbox("Pelvis_PFLRMD", [0,1,2,3,4], key="m_pelvis_pflrmd")
        pelvis_pflrmi_m = st.selectbox("Pelvis_PFLRMI", [0,1,2,3,4], key="m_pelvis_pflrmi")
        pelvis_pfcmd_m  = st.selectbox("Pelvis_PFCMD",  [0,1,2,3,4], key="m_pelvis_pfcmd")
        pelvis_pfcmi_m  = st.selectbox("Pelvis_PFCMI",  [0,1,2,3,4], key="m_pelvis_pfcmi")

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
st.markdown("Esta tabla muestra el coeficiente Kappa de Cohen para cada parámetro de asimetría, comparando la evaluación del veterinario y el modelo de IA. Un valor más alto indica mayor acuerdo entre ambos.")
st.dataframe(kappa_df)

st.markdown("Este gráfico de barras visualiza el nivel de acuerdo (Kappa de Cohen) entre el veterinario y el modelo de IA para cada parámetro de asimetría.")
st.bar_chart(kappa_df["Kappa"])

# Calculate Cohen's Kappa for Head and Pelvis separately
head_cols   = [c for c in param_cols if isinstance(c, str) and c.startswith("Cabeza_")]
pelvis_cols = [c for c in param_cols if isinstance(c, str) and c.startswith("Pelvis_")]

kappa_head = cohen_kappa_score(df_v[head_cols].values.flatten(), df_m[head_cols].values.flatten())
kappa_pelvis = cohen_kappa_score(df_v[pelvis_cols].values.flatten(), df_m[pelvis_cols].values.flatten())

# Calculate Cohen's Kappa for LR (Linea Recta) and C (Círculo) indicators separately
lr_cols = [c for c in param_cols if isinstance(c, str) and ("_LR" in c)]
circulo_cols = [c for c in param_cols if isinstance(c, str) and ("_C" in c and not "_LR" in c)]

kappa_lr = cohen_kappa_score(df_v[lr_cols].values.flatten(), df_m[lr_cols].values.flatten())
kappa_circulo = cohen_kappa_score(df_v[circulo_cols].values.flatten(), df_m[circulo_cols].values.flatten())

# 3.3 summary metrics
mean_kappa = kappa_df["Kappa"].mean()

st.subheader("Resumen Global")
mean_kappa = kappa_df["Kappa"].mean()
st.markdown("El valor mostrado representa el promedio del coeficiente Kappa de Cohen para todos los parámetros, brindando una visión global del nivel de acuerdo entre el veterinario y el modelo de IA.")

# Display Kappa Medio, Kappa Cabeza, and Kappa Pelvis side by side
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Kappa Medio", f"{mean_kappa:.2f}")
with col2:
    st.metric("Kappa Cabeza", f"{kappa_head:.2f}")
with col3:
    st.metric("Kappa Pelvis", f"{kappa_pelvis:.2f}")

# Display Kappa LR and Kappa Círculo side by side
col4, col5 = st.columns(2)
with col4:
    st.metric("Kappa LR", f"{kappa_lr:.2f}")
with col5:  
    st.metric("Kappa Círculo", f"{kappa_circulo:.2f}")

# ───────────────────────────────────────────────────────────────────────────────
# Análisis adicionales
st.markdown("---")
st.header("Análisis Adicional")

# 0. finding the max indicator for each horse
head_cols   = [c for c in param_cols if c.startswith("Cabeza_")]
pelvis_cols = [c for c in param_cols if c.startswith("Pelvis_")]


# ─────────────────────────────────────────────────────────────
st.subheader("Indicadores más afectados por caballo")
st.markdown("Esta sección muestra, para la población analizada, la frecuencia con la que cada parámetro de asimetría de cabeza y pelvis es el más afectado, separados para Veterinario y Modelo AI. Los gráficos de barras permiten comparar visualmente la prevalencia de cada indicador.")

# VETERINARIO
df_ind_vet = pd.DataFrame({
    "Max_Indicador_Cabeza": df_vet[head_cols].idxmax(axis=1),
    "Max_Indicador_Pelvis": df_vet[pelvis_cols].idxmax(axis=1)
}, index=df_vet["Caballo_ID"])
freq_cabeza_vet = df_ind_vet["Max_Indicador_Cabeza"].value_counts().sort_index()
freq_pelvis_vet = df_ind_vet["Max_Indicador_Pelvis"].value_counts().sort_index()

# MODELO AI
df_ind_ai = pd.DataFrame({
    "Max_Indicador_Cabeza": df_model[head_cols].idxmax(axis=1),
    "Max_Indicador_Pelvis": df_model[pelvis_cols].idxmax(axis=1)
}, index=df_model["Caballo_ID"])
freq_cabeza_ai = df_ind_ai["Max_Indicador_Cabeza"].value_counts().sort_index()
freq_pelvis_ai = df_ind_ai["Max_Indicador_Pelvis"].value_counts().sort_index()

def get_gradual_colorscale(n, colorscale="Viridis"):
    return {k: v for k, v in zip(range(n), sample_colorscale(colorscale, [i/(n-1) for i in range(n)]))}

# VET charts
color_cabeza_vet = get_gradual_colorscale(len(freq_cabeza_vet.index), "Viridis")
fig_cabeza_vet = px.bar(
    y=freq_cabeza_vet.index.tolist(),
    x=freq_cabeza_vet.values,
    color=freq_cabeza_vet.index.tolist(),
    orientation='h',
    color_discrete_map={k: color_cabeza_vet[i] for i, k in enumerate(freq_cabeza_vet.index.tolist())},
    labels={"y": "Indicador de Cabeza", "x": "Frecuencia"},
    title="Veterinario: Frecuencia de indicadores más afectados (Cabeza)"
)
color_pelvis_vet = get_gradual_colorscale(len(freq_pelvis_vet.index), "Viridis")
fig_pelvis_vet = px.bar(
    y=freq_pelvis_vet.index.tolist(),
    x=freq_pelvis_vet.values,
    color=freq_pelvis_vet.index.tolist(),
    orientation='h',
    color_discrete_map={k: color_pelvis_vet[i] for i, k in enumerate(freq_pelvis_vet.index.tolist())},
    labels={"y": "Indicador de Pelvis", "x": "Frecuencia"},
    title="Veterinario: Frecuencia de indicadores más afectados (Pelvis)"
)

# AI charts
color_cabeza_ai = get_gradual_colorscale(len(freq_cabeza_ai.index), "Viridis")
fig_cabeza_ai = px.bar(
    y=freq_cabeza_ai.index.tolist(),
    x=freq_cabeza_ai.values,
    color=freq_cabeza_ai.index.tolist(),
    orientation='h',
    color_discrete_map={k: color_cabeza_ai[i] for i, k in enumerate(freq_cabeza_ai.index.tolist())},
    labels={"y": "Indicador de Cabeza", "x": "Frecuencia"},
    title="Modelo AI: Frecuencia de indicadores más afectados (Cabeza)"
)
color_pelvis_ai = get_gradual_colorscale(len(freq_pelvis_ai.index), "Viridis")
fig_pelvis_ai = px.bar(
    y=freq_pelvis_ai.index.tolist(),
    x=freq_pelvis_ai.values,
    color=freq_pelvis_ai.index.tolist(),
    orientation='h',
    color_discrete_map={k: color_pelvis_ai[i] for i, k in enumerate(freq_pelvis_ai.index.tolist())},
    labels={"y": "Indicador de Pelvis", "x": "Frecuencia"},
    title="Modelo AI: Frecuencia de indicadores más afectados (Pelvis)"
)

col_vet_cab, col_vet_pel = st.columns(2)
with col_vet_cab:
    st.plotly_chart(fig_cabeza_vet, use_container_width=True)
with col_vet_pel:
    st.plotly_chart(fig_pelvis_vet, use_container_width=True)

col_ai_cab, col_ai_pel = st.columns(2)
with col_ai_cab:
    st.plotly_chart(fig_cabeza_ai, use_container_width=True)
with col_ai_pel:
    st.plotly_chart(fig_pelvis_ai, use_container_width=True)

# 1.1. race vs sex vs age

# ───────────────────────────────────────────────────────────────────────────────
# Frecuencia de miembros más afectados por raza y sexo
st.subheader("Frecuencia de miembros más afectados por raza y por sexo")
st.markdown("Esta sección muestra la frecuencia con la que cada miembro (cabeza o pelvis) es el más afectado, agrupado por raza y por sexo, según la evaluación veterinaria. Los gráficos permiten comparar visualmente la prevalencia de cada miembro en cada grupo.")

# Determinar miembro más afectado por caballo (cabeza vs pelvis)
def miembro_mas_afectado(row):
    cabeza_max = row[head_cols].max()
    pelvis_max = row[pelvis_cols].max()
    if cabeza_max > pelvis_max:
        return "Cabeza"
    elif pelvis_max > cabeza_max:
        return "Pelvis"
    else:
        return "Empate"

df_vet["Miembro_Mas_Afectado"] = df_vet.apply(miembro_mas_afectado, axis=1)



# Frecuencia de miembro más afectado por raza (Veterinario)
freq_raza_grouped = df_vet.groupby(["Raza", "Miembro_Mas_Afectado"]).size().reset_index(name="Frecuencia")
fig_raza_grouped = px.bar(
    freq_raza_grouped,
    x="Raza",
    y="Frecuencia",
    color="Miembro_Mas_Afectado",
    barmode="group",
    color_discrete_map={"Cabeza": "#4682B4", "Pelvis": "#228B22", "Empate": "#FFD700"},
    labels={"Raza": "Raza", "Frecuencia": "Frecuencia", "Miembro_Mas_Afectado": "Miembro"},
    title="Frecuencia de miembro más afectado por raza (Veterinario)"
)
st.plotly_chart(fig_raza_grouped, use_container_width=True)

# Frecuencia de miembro más afectado por raza (Modelo AI)
def miembro_mas_afectado_ai(row):
    cabeza_max = row[head_cols].max()
    pelvis_max = row[pelvis_cols].max()
    if cabeza_max > pelvis_max:
        return "Cabeza"
    elif pelvis_max > cabeza_max:
        return "Pelvis"
    else:
        return "Empate"

df_model["Miembro_Mas_Afectado"] = df_model.apply(miembro_mas_afectado_ai, axis=1)
freq_raza_grouped_ai = df_model.groupby(["Raza", "Miembro_Mas_Afectado"]).size().reset_index(name="Frecuencia")
fig_raza_grouped_ai = px.bar(
    freq_raza_grouped_ai,
    x="Raza",
    y="Frecuencia",
    color="Miembro_Mas_Afectado",
    barmode="group",
    color_discrete_map={"Cabeza": "#4682B4", "Pelvis": "#228B22", "Empate": "#FFD700"},
    labels={"Raza": "Raza", "Frecuencia": "Frecuencia", "Miembro_Mas_Afectado": "Miembro"},
    title="Frecuencia de miembro más afectado por raza (Modelo AI)"
)
st.plotly_chart(fig_raza_grouped_ai, use_container_width=True)


# Frecuencia de miembro más afectado por sexo (Veterinario)
freq_sexo_grouped_vet = df_vet.groupby(["Sexo", "Miembro_Mas_Afectado"]).size().reset_index(name="Frecuencia")
fig_sexo_vet = px.bar(
    freq_sexo_grouped_vet,
    x="Sexo",
    y="Frecuencia",
    color="Miembro_Mas_Afectado",
    barmode="group",
    color_discrete_map={"Cabeza": "#4682B4", "Pelvis": "#228B22", "Empate": "#FFD700"},
    labels={"Sexo": "Sexo", "Frecuencia": "Frecuencia", "Miembro_Mas_Afectado": "Miembro"},
    title="Frecuencia de miembro más afectado por sexo (Veterinario)"
)

# Frecuencia de miembro más afectado por sexo (Modelo AI)
freq_sexo_grouped_ai = df_model.groupby(["Sexo", "Miembro_Mas_Afectado"]).size().reset_index(name="Frecuencia")
fig_sexo_ai = px.bar(
    freq_sexo_grouped_ai,
    x="Sexo",
    y="Frecuencia",
    color="Miembro_Mas_Afectado",
    barmode="group",
    color_discrete_map={"Cabeza": "#4682B4", "Pelvis": "#228B22", "Empate": "#FFD700"},
    labels={"Sexo": "Sexo", "Frecuencia": "Frecuencia", "Miembro_Mas_Afectado": "Miembro"},
    title="Frecuencia de miembro más afectado por sexo (Modelo AI)"
)

col_vet_sexo, col_ai_sexo = st.columns(2)
with col_vet_sexo:
    st.plotly_chart(fig_sexo_vet, use_container_width=True)
with col_ai_sexo:
    st.plotly_chart(fig_sexo_ai, use_container_width=True)

# 2. frequency of qualitative variables
st.subheader("Frecuencia de variables cualitativas")
st.markdown("Estas tablas y gráficos muestran la frecuencia de aparición de cada valor en las variables cualitativas (raza, sexo, análisis clínico y condición corporal) en la base de datos veterinaria.")
qual_cols = ["Raza","Sexo","Analisis_clinico","Condicion_Corporal"]
for col in qual_cols:
    st.markdown(f"**{col}**")
    vc = df_vet[col].value_counts()
    st.dataframe(vc.to_frame("Frecuencia"))
    st.bar_chart(vc)

# augments distribution after flexion
st.markdown("---")
st.subheader("Distribución de aumentos tras flexión (Veterinario)")
st.markdown("Esta tabla y gráfico muestran la distribución de los cambios en los parámetros de asimetría tras la flexión, clasificando los cambios en categorías de incremento ('+', '++', '+++', '++++'), sin cambio ('='), y decremento ('-', '--', '---', '----'). Permite visualizar cuántos caballos presentan cada nivel de cambio en cada indicador según la evaluación veterinaria.")

before_cols = [
    c for c in df_vet.columns 
    if (c.startswith("Cabeza_") or c.startswith("Pelvis_")) 
       and not c.startswith(("Cabeza_P","Pelvis_P"))
]
# find all before/after pairs based on your column naming
pairs = {}
for c in df_vet.columns:
    if c.startswith("Cabeza_") or c.startswith("Pelvis_"):
        # only non-PF columns as "before"
        if not ("PFL" in c or "PFC" in c):
            prefix, indicator = c.split("_", 1)
            # find any after column that starts with prefix + "_PF" and ends with indicator
            for after in df_vet.columns:
                if after.startswith(f"{prefix}_PF") and after.endswith(indicator):
                    pairs[c] = after
                    break  # only take the first match

# add bins for decrements as well
# order: '---', '--', '-', '=', '+', '++', '+++', '++++'
diff_bins = ["----", "---", "--", "-", "=", "+", "++", "+++", "++++"]
records = {}
for before, after in pairs.items():
    diff = df_vet[after] - df_vet[before]
    prefix, indicator = before.split("_", 1)
    row_name = f"{prefix}_{indicator}"
    counts = {
        "----":int((diff <= -4).sum()),
        "---": int((diff == -3).sum()),
        "--":  int((diff == -2).sum()),
        "-":   int((diff == -1).sum()),
        "=":   int((diff == 0).sum()),
        "+":   int((diff == 1).sum()),
        "++":  int((diff == 2).sum()),
        "+++": int((diff == 3).sum()),
        "++++":int((diff >= 4).sum()),
    }
    records[row_name] = counts

df_diff = pd.DataFrame.from_dict(records, orient="index", columns=diff_bins)
st.dataframe(df_diff)

# Custom bar chart for augments distribution (Vet)
color_map = {
    "----": "#8B0000", # dark red
    "---": "#B22222", # firebrick
    "--": "#DC143C", # crimson
    "-": "#FF6347", # tomato
    "=": "#4682B4", # steel blue
    "+": "#32CD32", # lime green
    "++": "#228B22", # forest green
    "+++": "#006400", # dark green
    "++++": "#2E8B57", # sea green
}

vet_counts = df_diff.sum(axis=0)
fig_vet = px.bar(
    x=vet_counts.index,
    y=vet_counts.values,
    color=vet_counts.index,
    color_discrete_map=color_map,
    labels={"x": "Categoría", "y": "Frecuencia"},
    title="Distribución total de aumentos tras flexión (Veterinario)"
)
st.plotly_chart(fig_vet, use_container_width=True)

# --- ai model version ---
st.subheader("Distribución de aumentos tras flexión (Modelo AI)")
st.markdown("Esta tabla y gráfico muestran la distribución de los cambios en los parámetros de asimetría tras la flexión, clasificando los cambios en categorías de incremento ('+', '++', '+++', '++++'), sin cambio ('='), y decremento ('-', '--', '---', '----'). Permite visualizar cuántos caballos presentan cada nivel de cambio en cada indicador según el modelo de IA.")

# find all before/after pairs based on your column naming
pairs_model = {}
for c in df_model.columns:
    if c.startswith("Cabeza_") or c.startswith("Pelvis_"):
        # only non-PF columns as "before"
        if not ("PFL" in c or "PFC" in c):
            prefix, indicator = c.split("_", 1)
            # find any after column that starts with prefix + "_PF" and ends with indicator
            for after in df_model.columns:
                if after.startswith(f"{prefix}_PF") and after.endswith(indicator):
                    pairs_model[c] = after
                    break  # only take the first match

# add bins for decrements as well
records_model = {}
for before, after in pairs_model.items():
    diff = df_model[after] - df_model[before]
    prefix, indicator = before.split("_", 1)
    row_name = f"{prefix}_{indicator}"
    counts = {
        "----":int((diff <= -4).sum()),
        "---": int((diff == -3).sum()),
        "--":  int((diff == -2).sum()),
        "-":   int((diff == -1).sum()),
        "=":   int((diff == 0).sum()),
        "+":   int((diff == 1).sum()),
        "++":  int((diff == 2).sum()),
        "+++": int((diff == 3).sum()),
        "++++":int((diff >= 4).sum()),
    }
    records_model[row_name] = counts

df_diff_model = pd.DataFrame.from_dict(records_model, orient="index", columns=diff_bins)
st.dataframe(df_diff_model)

# Custom bar chart for augments distribution (AI)
ai_counts = df_diff_model.sum(axis=0)
fig_ai = px.bar(
    x=ai_counts.index,
    y=ai_counts.values,
    color=ai_counts.index,
    color_discrete_map=color_map,
    labels={"x": "Categoría", "y": "Frecuencia"},
    title="Distribución total de aumentos tras flexión (Modelo AI)"
)
st.plotly_chart(fig_ai, use_container_width=True)

# Resumen estadístico de variables cuantitativas
st.subheader("Resumen estadístico de Edad y Condición Corporal")
st.markdown("Esta tabla muestra la media, error estándar, desviación estándar, valor mínimo y máximo para las variables cuantitativas Edad y Condición Corporal en la base de datos veterinaria.")

def resumen_stats(serie):
    return pd.Series({
        "Media": np.mean(serie),
        "E.E": np.std(serie, ddof=1) / np.sqrt(len(serie)),
        "D.E": np.std(serie, ddof=1),
        "Mín.": np.min(serie),
        "Máx.": np.max(serie)
    })

# Convert Condicion Corporal to numeric if needed
cond_col = df_vet["Condicion_Corporal"].replace({',':'.'}, regex=True)
cond_col = pd.to_numeric(cond_col, errors='coerce')

stats_df = pd.DataFrame({
    "Edad": resumen_stats(df_vet["Edad"]),
    "Condición corporal": resumen_stats(cond_col)
}).T

st.dataframe(stats_df)
st.markdown("E.E = Error estándar; D.E = Desviación estándar; Mín. = Mínimo; Máx. = Máximo")
st.markdown("---")

# FINAL: Bar charts for claudication counts per grade (1-5) for Vet and AI
st.header("Resumen de calificaciones de claudicación (1-5)")
st.markdown("Estos gráficos muestran la cantidad de veces que el veterinario y la IA asignaron cada grado de claudicación (1-5) en cualquier indicador.")

# All indicator columns (head + pelvis)
all_indic_cols = [c for c in df_vet.columns if c.startswith("Cabeza_") or c.startswith("Pelvis_")]

# Flatten all grades for vet and AI
vet_grades = df_vet[all_indic_cols].values.flatten()
ai_grades = df_model[all_indic_cols].values.flatten()

# Count occurrences of grades 1-5 (ignore 0 if present)
vet_grade_counts = pd.Series(vet_grades).value_counts().sort_index()
ai_grade_counts = pd.Series(ai_grades).value_counts().sort_index()

# Only show grades 1-5

claud_colors = {
    1: "#8B0000",   # rojo oscuro
    2: "#FF6347",  # tomate
    3: "#FFD700",  # dorado
    4: "#4682B4",  # azul acero
    5: "#228B22",  # verde bosque
}

grades = [1,2,3,4,5]
vet_grade_counts = vet_grade_counts.reindex(grades, fill_value=0)
ai_grade_counts = ai_grade_counts.reindex(grades, fill_value=0)

fig_vet_claud = px.bar(
    x=vet_grade_counts.index,
    y=vet_grade_counts.values,
    color=vet_grade_counts.index,
    color_discrete_map=claud_colors,
    labels={"x": "Grado de claudicación", "y": "Frecuencia"},
    title="Veterinario: cantidad de veces que se asignó cada grado"
)
fig_ai_claud = px.bar(
    x=ai_grade_counts.index,
    y=ai_grade_counts.values,
    color=ai_grade_counts.index,
    color_discrete_map=claud_colors,
    labels={"x": "Grado de claudicación", "y": "Frecuencia"},
    title="Modelo AI: cantidad de veces que se asignó cada grado"
)

col_vet, col_ai = st.columns(2)
with col_vet:
    st.plotly_chart(fig_vet_claud, use_container_width=True)
with col_ai:
    st.plotly_chart(fig_ai_claud, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Cantidad de alteraciones por cabeza vs pelvis (Vet y AI)
st.markdown("---")
st.header("Cantidad de alteraciones: Cabeza vs Pelvis")
st.markdown("Estos gráficos muestran la cantidad total de alteraciones (valor >= 1) en todos los indicadores de cabeza y pelvis, separados para Veterinario y Modelo AI. Cada barra representa el total de veces que se detectó alguna alteración en los indicadores correspondientes.")

# Definir columnas de cabeza y pelvis
head_cols = [c for c in all_indic_cols if c.startswith("Cabeza_")]
pelvis_cols = [c for c in all_indic_cols if c.startswith("Pelvis_")]

# Veterinario: contar alteraciones (valor >=1) por cabeza y pelvis
vet_head_alter = (df_vet[head_cols] >= 1).sum().sum()
vet_pelvis_alter = (df_vet[pelvis_cols] >= 1).sum().sum()

# Modelo AI: contar alteraciones (valor >=1) por cabeza y pelvis
ai_head_alter = (df_model[head_cols] >= 1).sum().sum()
ai_pelvis_alter = (df_model[pelvis_cols] >= 1).sum().sum()

# Dataframes para graficar
vet_alter_df = pd.DataFrame({
    "Zona": ["Cabeza", "Pelvis"],
    "Alteraciones": [vet_head_alter, vet_pelvis_alter]
})
ai_alter_df = pd.DataFrame({
    "Zona": ["Cabeza", "Pelvis"],
    "Alteraciones": [ai_head_alter, ai_pelvis_alter]
})

# Colores para barras
zona_colors = {"Cabeza": "#4682B4", "Pelvis": "#228B22"}

fig_vet_alter = px.bar(
    vet_alter_df,
    x="Zona",
    y="Alteraciones",
    color="Zona",
    color_discrete_map=zona_colors,
    labels={"Zona": "Zona", "Alteraciones": "Cantidad de alteraciones"},
    title="Veterinario: cantidad total de alteraciones (valor >= 1)"
)
fig_ai_alter = px.bar(
    ai_alter_df,
    x="Zona",
    y="Alteraciones",
    color="Zona",
    color_discrete_map=zona_colors,
    labels={"Zona": "Zona", "Alteraciones": "Cantidad de alteraciones"},
    title="Modelo AI: cantidad total de alteraciones (valor >= 1)"
)

col_vet_alter, col_ai_alter = st.columns(2)
with col_vet_alter:
    st.plotly_chart(fig_vet_alter, use_container_width=True)
with col_ai_alter:
    st.plotly_chart(fig_ai_alter, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Cantidad de alteraciones por cabeza/pelvis, divididas por raza y sexo (Vet y AI)
st.markdown("---")
st.header("Cantidad de alteraciones por Cabeza/Pelvis, divididas por Raza y Sexo")
st.markdown("Estos gráficos muestran la cantidad total de alteraciones (valor >= 1) en todos los indicadores de cabeza y pelvis, agrupadas por raza y por sexo, para Veterinario y Modelo AI.")

# Por Raza
## Veterinario
vet_alter_raza = []
for raza, group in df_vet.groupby("Raza"):
    head_alter = (group[head_cols] >= 1).sum().sum()
    pelvis_alter = (group[pelvis_cols] >= 1).sum().sum()
    vet_alter_raza.append({"Raza": raza, "Zona": "Cabeza", "Alteraciones": head_alter})
    vet_alter_raza.append({"Raza": raza, "Zona": "Pelvis", "Alteraciones": pelvis_alter})
vet_alter_raza_df = pd.DataFrame(vet_alter_raza)

fig_vet_alter_raza = px.bar(
    vet_alter_raza_df,
    x="Raza",
    y="Alteraciones",
    color="Zona",
    barmode="group",
    color_discrete_map=zona_colors,
    labels={"Raza": "Raza", "Alteraciones": "Cantidad de alteraciones"},
    title="Veterinario: alteraciones por raza"
)

## Modelo AI
ai_alter_raza = []
for raza, group in df_model.groupby("Raza"):
    head_alter = (group[head_cols] >= 1).sum().sum()
    pelvis_alter = (group[pelvis_cols] >= 1).sum().sum()
    ai_alter_raza.append({"Raza": raza, "Zona": "Cabeza", "Alteraciones": head_alter})
    ai_alter_raza.append({"Raza": raza, "Zona": "Pelvis", "Alteraciones": pelvis_alter})
ai_alter_raza_df = pd.DataFrame(ai_alter_raza)

fig_ai_alter_raza = px.bar(
    ai_alter_raza_df,
    x="Raza",
    y="Alteraciones",
    color="Zona",
    barmode="group",
    color_discrete_map=zona_colors,
    labels={"Raza": "Raza", "Alteraciones": "Cantidad de alteraciones"},
    title="Modelo AI: alteraciones por raza"
)

col_vet_raza, col_ai_raza = st.columns(2)
with col_vet_raza:
    st.plotly_chart(fig_vet_alter_raza, use_container_width=True)
with col_ai_raza:
    st.plotly_chart(fig_ai_alter_raza, use_container_width=True)

# Por Sexo
## Veterinario
vet_alter_sexo = []
for sexo, group in df_vet.groupby("Sexo"):
    head_alter = (group[head_cols] >= 1).sum().sum()
    pelvis_alter = (group[pelvis_cols] >= 1).sum().sum()
    vet_alter_sexo.append({"Sexo": sexo, "Zona": "Cabeza", "Alteraciones": head_alter})
    vet_alter_sexo.append({"Sexo": sexo, "Zona": "Pelvis", "Alteraciones": pelvis_alter})
vet_alter_sexo_df = pd.DataFrame(vet_alter_sexo)

fig_vet_alter_sexo = px.bar(
    vet_alter_sexo_df,
    x="Sexo",
    y="Alteraciones",
    color="Zona",
    barmode="group",
    color_discrete_map=zona_colors,
    labels={"Sexo": "Sexo", "Alteraciones": "Cantidad de alteraciones"},
    title="Veterinario: alteraciones por sexo"
)

## Modelo AI
ai_alter_sexo = []
for sexo, group in df_model.groupby("Sexo"):
    head_alter = (group[head_cols] >= 1).sum().sum()
    pelvis_alter = (group[pelvis_cols] >= 1).sum().sum()
    ai_alter_sexo.append({"Sexo": sexo, "Zona": "Cabeza", "Alteraciones": head_alter})
    ai_alter_sexo.append({"Sexo": sexo, "Zona": "Pelvis", "Alteraciones": pelvis_alter})
ai_alter_sexo_df = pd.DataFrame(ai_alter_sexo)

fig_ai_alter_sexo = px.bar(
    ai_alter_sexo_df,
    x="Sexo",
    y="Alteraciones",
    color="Zona",
    barmode="group",
    color_discrete_map=zona_colors,
    labels={"Sexo": "Sexo", "Alteraciones": "Cantidad de alteraciones"},
    title="Modelo AI: alteraciones por sexo"
)

col_vet_sexo_alter, col_ai_sexo_alter = st.columns(2)
with col_vet_sexo_alter:
    st.plotly_chart(fig_vet_alter_sexo, use_container_width=True)
with col_ai_sexo_alter:
    st.plotly_chart(fig_ai_alter_sexo, use_container_width=True)

