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
import plotly.graph_objects as go

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

st.markdown("Este gráfico de barras visualiza el nivel de acuerdo (Kappa de Cohen) entre el veterinario y el modelo de IA para cada parámetro de asimetría, ordenado de mayor a menor.")
# Ordenar el DataFrame por Kappa descendente antes de graficar
kappa_sorted = kappa_df.sort_values("Kappa", ascending=False)
fig_kappa_sorted = go.Figure(go.Bar(
    x=kappa_sorted.index,
    y=kappa_sorted["Kappa"],
    marker_color="#4682B4",
    width=0.5  # Reduce bar width for compactness
))
fig_kappa_sorted.update_layout(
    xaxis_title="Parámetro",
    yaxis_title="Kappa de Cohen",
    title="Nivel de acuerdo (Kappa de Cohen) ordenado de mayor a menor",
    xaxis_tickangle=-45,
    width=700  # Reduce overall figure width
)
st.plotly_chart(fig_kappa_sorted, use_container_width=False)

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
st.markdown("---")
st.header("Cantidad de alteraciones: Cabeza vs Pelvis")
st.markdown("Estos gráficos muestran la cantidad total de alteraciones (valor >= 1) en todos los indicadores de cabeza y pelvis, separados para Veterinario y Modelo AI. Cada barra representa el total de veces que se detectó alguna alteración en los indicadores correspondientes.")

head_cols = [c for c in all_indic_cols if c.startswith("Cabeza_")]
pelvis_cols = [c for c in all_indic_cols if c.startswith("Pelvis_")]

vet_head_alter = (df_vet[head_cols] >= 1).sum().sum()
vet_pelvis_alter = (df_vet[pelvis_cols] >= 1).sum().sum()

ai_head_alter = (df_model[head_cols] >= 1).sum().sum()
ai_pelvis_alter = (df_model[pelvis_cols] >= 1).sum().sum()

# Dataframes para graficar
vet_alter_df = pd.DataFrame({
    "Zona": ["Miembro Anterior", "Miembro Posterior"],
    "Alteraciones": [vet_head_alter, vet_pelvis_alter]
})
ai_alter_df = pd.DataFrame({
    "Zona": ["Miembro Anterior", "Miembro Posterior"],
    "Alteraciones": [ai_head_alter, ai_pelvis_alter]
})

# Colores para barras
zona_colors = {"Miembro Anterior": "#4682B4", "Miembro Posterior": "#228B22"}

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
    vet_alter_raza.append({"Raza": raza, "Zona": "Miembro Anterior", "Alteraciones": head_alter})
    vet_alter_raza.append({"Raza": raza, "Zona": "Miembro Posterior", "Alteraciones": pelvis_alter})
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
    ai_alter_raza.append({"Raza": raza, "Zona": "Miembro Anterior", "Alteraciones": head_alter})
    ai_alter_raza.append({"Raza": raza, "Zona": "Miembro Posterior", "Alteraciones": pelvis_alter})
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
    vet_alter_sexo.append({"Sexo": sexo, "Zona": "Miembro Anterior", "Alteraciones": head_alter})
    vet_alter_sexo.append({"Sexo": sexo, "Zona": "Miembro Posterior", "Alteraciones": pelvis_alter})
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
    ai_alter_sexo.append({"Sexo": sexo, "Zona": "Miembro Anterior", "Alteraciones": head_alter})
    ai_alter_sexo.append({"Sexo": sexo, "Zona": "Miembro Posterior", "Alteraciones": pelvis_alter})
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

# ─────────────────────────────────────────────────────────────
st.markdown("---")

col_text, col_plot = st.columns([1, 2]) 

with col_text:
    st.header("Concordancia de Cohen Promedio")
    st.markdown("""
    Este gráfico compara la **concordancia promedio (0 a 1)** entre el 
    veterinario y el modelo de IA para:
    - **Miembros Anteriores (Cabeza)**
    - **Miembros Posteriores (Pelvis)**
    """)

with col_plot:
    kappa_avg_df = pd.DataFrame({
        "Zona": ["Miembros Anteriores", "Miembros Posteriores"],
        "Kappa Promedio": [kappa_head, kappa_pelvis]
    })

    kappa_colors = {
        "Miembros Anteriores": "#4682B4",
        "Miembros Posteriores": "#228B22"
    }

    fig_kappa_avg = px.bar(
        kappa_avg_df,
        x="Zona",
        y="Kappa Promedio",
        color="Zona",
        color_discrete_map=kappa_colors,
        labels={"Zona": "Zona", "Kappa Promedio": "Cohen's Kappa (Promedio)"},
        title="Concordancia (0 a 1)"
    )

    st.plotly_chart(fig_kappa_avg, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# True Positive Rate and True Negative Rate (Vet = true, AI = pred)

st.markdown("---")
st.header("Tasas de VP, VN, FP y FN (Modelo AI vs Veterinario)")
st.markdown("""
Se calcula la tasa de verdaderos positivos (**VP**, sensibilidad) y verdaderos negativos (**VN**, especificidad), así como los falsos positivos (**FP**) y falsos negativos (**FN**), considerando como referencia la evaluación del veterinario y como predicción la del modelo AI. Se consideran todos los indicadores de cabeza y pelvis juntos.

- **Positivo:** Se considera alteración (valor ≥ 1)
- **Negativo:** Se considera normalidad (valor = 0)

**Definiciones:**
- **VP (Verdaderos positivos):** El modelo AI detecta alteración y el veterinario también.
- **VN (Verdaderos negativos):** El modelo AI detecta normalidad y el veterinario también.
- **FP (Falsos positivos):** El modelo AI detecta alteración pero el veterinario no.
- **FN (Falsos negativos):** El modelo AI detecta normalidad pero el veterinario detecta alteración.

---
### Fórmulas utilizadas

<div style="font-size: 1.1em">
<b>VP</b> = Número de casos donde <b>AI ≥ 1</b> y <b>Vet ≥ 1</b><br>
<b>VN</b> = Número de casos donde <b>AI = 0</b> y <b>Vet = 0</b><br>
<b>FP</b> = Número de casos donde <b>AI ≥ 1</b> y <b>Vet = 0</b><br>
<b>FN</b> = Número de casos donde <b>AI = 0</b> y <b>Vet ≥ 1</b><br>
<br>
<b>Tasa de VP (Sensibilidad)</b> = VP / (VP + FN)<br>
<b>Tasa de VN (Especificidad)</b> = VN / (VN + FP)
</div>

Donde "casos" se refiere a cada celda de todos los indicadores de cabeza y pelvis.<br>
<br>
<b>AI</b> = valor del modelo IA para ese indicador<br>
<b>Vet</b> = valor del veterinario para ese indicador
""", unsafe_allow_html=True)


# Fórmulas detalladas y explicación
st.markdown("""
<div style='font-size:1.1em'>
<b>Fórmulas detalladas para cada indicador:</b><br>
Para cada indicador <b>i</b>:
<ul>
<li><b>VP<sub>i</sub></b> = Número de casos donde <b>AI<sub>i</sub> ≥ 1</b> y <b>Vet<sub>i</sub> ≥ 1</b></li>
<li><b>VN<sub>i</sub></b> = Número de casos donde <b>AI<sub>i</sub> = 0</b> y <b>Vet<sub>i</sub> = 0</b></li>
<li><b>FP<sub>i</sub></b> = Número de casos donde <b>AI<sub>i</sub> ≥ 1</b> y <b>Vet<sub>i</sub> = 0</b></li>
<li><b>FN<sub>i</sub></b> = Número de casos donde <b>AI<sub>i</sub> = 0</b> y <b>Vet<sub>i</sub> ≥ 1</b></li>
</ul>
<b>Tasa de VP<sub>i</sub> (Sensibilidad)</b> = VP<sub>i</sub> / (VP<sub>i</sub> + FN<sub>i</sub>)<br>
<b>Tasa de VN<sub>i</sub> (Especificidad)</b> = VN<sub>i</sub> / (VN<sub>i</sub> + FP<sub>i</sub>)<br>
<br>
Donde cada indicador se calcula por columna, es decir, por parámetro de asimetría.<br>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<b>¿Qué muestra la siguiente tabla?</b><br>
Para cada parámetro de asimetría (columna), se muestran los valores de VP, VN, FP, FN, la tasa de VP (sensibilidad) y la tasa de VN (especificidad), calculados individualmente.<br>
<br>
Esto permite analizar el desempeño del modelo AI respecto al veterinario para cada indicador específico.
""", unsafe_allow_html=True)

# Seleccionar columnas de indicadores
indic_cols = [c for c in df_vet.columns if c.startswith("Cabeza_") or c.startswith("Pelvis_")]


# Calcular métricas por indicador
per_indicator = []
for col in indic_cols:
    vet_col = df_vet[col]
    ai_col = df_model[col]
    y_true = (vet_col >= 1).astype(int)
    y_pred = (ai_col >= 1).astype(int)
    VP = ((y_true == 1) & (y_pred == 1)).sum()
    VN = ((y_true == 0) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()
    TPR = VP / (VP + FN) if (VP + FN) > 0 else float('nan')
    TNR = VN / (VN + FP) if (VN + FP) > 0 else float('nan')
    per_indicator.append({
        "Indicador": col,
        "VP": VP,
        "VN": VN,
        "FP": FP,
        "FN": FN,
        "Tasa VP (Sensibilidad)": TPR,
        "Tasa VN (Especificidad)": TNR
    })

per_indicator_df = pd.DataFrame(per_indicator)
st.dataframe(per_indicator_df.set_index("Indicador"))

# Agrupar tasas por Cabeza y Pelvis
head_cols = [c for c in indic_cols if c.startswith("Cabeza_")]
pelvis_cols = [c for c in indic_cols if c.startswith("Pelvis_")]

def group_sens_spec(cols):
    # Agrupa los valores de VP, VN, FP, FN para el grupo dado
    VP = VN = FP = FN = 0
    for col in cols:
        row = per_indicator_df[per_indicator_df["Indicador"] == col]
        if not row.empty:
            VP += row["VP"].values[0]
            VN += row["VN"].values[0]
            FP += row["FP"].values[0]
            FN += row["FN"].values[0]
    TPR = VP / (VP + FN) if (VP + FN) > 0 else float('nan')
    TNR = VN / (VN + FP) if (VN + FP) > 0 else float('nan')
    return VP, VN, FP, FN, TPR, TNR

VP_head, VN_head, FP_head, FN_head, TPR_head, TNR_head = group_sens_spec(head_cols)
VP_pelvis, VN_pelvis, FP_pelvis, FN_pelvis, TPR_pelvis, TNR_pelvis = group_sens_spec(pelvis_cols)


st.markdown("""
<b>Resumen agrupado por zona:</b><br>
<ul>
<li><b>Miembros Anteriores (Cabeza):</b> Tasa de VP (Sensibilidad): <b>{:.2%}</b> &nbsp;|&nbsp; Tasa de VN (Especificidad): <b>{:.2%}</b> &nbsp;|&nbsp; VP: {} &nbsp;|&nbsp; VN: {} &nbsp;|&nbsp; FP: {} &nbsp;|&nbsp; FN: {}</li>
<li><b>Miembros Posteriores (Pelvis):</b> Tasa de VP (Sensibilidad): <b>{:.2%}</b> &nbsp;|&nbsp; Tasa de VN (Especificidad): <b>{:.2%}</b> &nbsp;|&nbsp; VP: {} &nbsp;|&nbsp; VN: {} &nbsp;|&nbsp; FP: {} &nbsp;|&nbsp; FN: {}</li>
</ul>
""".format(
    TPR_head, TNR_head, VP_head, VN_head, FP_head, FN_head,
    TPR_pelvis, TNR_pelvis, VP_pelvis, VN_pelvis, FP_pelvis, FN_pelvis
), unsafe_allow_html=True)

# Gráfico de barras para comparar sensibilidad y especificidad agrupadas
sens_spec_df = pd.DataFrame({
    "Zona": ["Miembros Anteriores (Cabeza)", "Miembros Posteriores (Pelvis)"],
    "Sensibilidad (Tasa VP)": [TPR_head, TPR_pelvis],
    "Especificidad (Tasa VN)": [TNR_head, TNR_pelvis]
})

fig_sens_spec = go.Figure()
fig_sens_spec.add_bar(
    x=sens_spec_df["Zona"],
    y=sens_spec_df["Sensibilidad (Tasa VP)"],
    name="Sensibilidad (Tasa VP)",
    marker_color="#4682B4"
)
fig_sens_spec.add_bar(
    x=sens_spec_df["Zona"],
    y=sens_spec_df["Especificidad (Tasa VN)"],
    name="Especificidad (Tasa VN)",
    marker_color="#228B22"
)
fig_sens_spec.update_layout(
    barmode="group",
    title="Comparación de Sensibilidad y Especificidad agrupadas por zona",
    yaxis_title="Valor",
    xaxis_title="Zona",
    yaxis_tickformat=",.0%",
    width=700
)
st.plotly_chart(fig_sens_spec, use_container_width=False)

# Calcular métricas globales (todas las celdas)
y_true_all = (df_vet[indic_cols].values.flatten() >= 1).astype(int)
y_pred_all = (df_model[indic_cols].values.flatten() >= 1).astype(int)
TP_all = ((y_true_all == 1) & (y_pred_all == 1)).sum()
TN_all = ((y_true_all == 0) & (y_pred_all == 0)).sum()
FP_all = ((y_true_all == 0) & (y_pred_all == 1)).sum()
FN_all = ((y_true_all == 1) & (y_pred_all == 0)).sum()
TPR_all = TP_all / (TP_all + FN_all) if (TP_all + FN_all) > 0 else float('nan')
TNR_all = TN_all / (TN_all + FP_all) if (TN_all + FP_all) > 0 else float('nan')

col_vp, col_vn = st.columns(2)
with col_vp:
    st.metric("Tasa de VP (Sensibilidad, global)", f"{TPR_all:.2%}")
    st.markdown(f"VP: {TP_all}, FN: {FN_all}")
with col_vn:
    st.metric("Tasa de VN (Especificidad, global)", f"{TNR_all:.2%}")
    st.markdown(f"VN: {TN_all}, FP: {FP_all}")

# app.py
# Usage: streamlit run app.py

st.set_page_config(page_title="Fisher Test: MV vs SGAH", layout="wide")
st.title("Fisher’s Exact Test — MV Total vs SGAH Total")

st.markdown(
    """
For each row/category, we compare **MV Total** vs **SGAH Total** with a 2×2 Fisher's exact test:
    | In Category | Not in Category |
MV | a | b |
SGAH | c | d |

H₀: the in-category proportion is the same for MV and SGAH.
"""
)

# -----------------------------------------------------------------------------
# Built-in data from your tables (you can edit these numbers here if needed)
# -----------------------------------------------------------------------------
data_sources = {
    "Breeds (Raza)": pd.DataFrame(
        {
            "Category": [
                "Iberoamericana",
                "Española",
                "Cuarto de milla",
                "Warmblood",
                "Frisón",
                "Criollo",
                "Paso costarricense",
                "Warlander",
            ],
            # MV Total per breed (sums to 289)
            "MV_in": [109, 64, 41, 38, 15, 14, 8, 0],
            # SGAH Total per breed (sums to 388)
            "SGAH_in": [139, 103, 57, 32, 20, 20, 9, 8],
        }
    ),
    "Sex (Sexo)": pd.DataFrame(
        {
            "Category": ["Hembras", "Machos"],
            "MV_in": [126, 163],   # sums to 289
            "SGAH_in": [169, 219], # sums to 388
        }
    ),
}

choice = st.selectbox("Choose dataset", list(data_sources.keys()))
df_in = data_sources[choice].copy()

# -----------------------------------------------------------------------------
# Totals (from your tables). Adjust here if your overall totals change.
# -----------------------------------------------------------------------------
MV_total = 289
SGAH_total = 388

c1, c2 = st.columns(2)
with c1:
    st.metric("MV Total (overall)", MV_total)
with c2:
    st.metric("SGAH Total (overall)", SGAH_total)

# -----------------------------------------------------------------------------
# Compute Fisher per category
# -----------------------------------------------------------------------------
results = []
for _, row in df_in.iterrows():
    cat = row["Category"]
    a = int(row["MV_in"])     # MV in-category
    c = int(row["SGAH_in"])   # SGAH in-category
    b = MV_total - a          # MV not-in-category
    d = SGAH_total - c        # SGAH not-in-category

    OR, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")

    results.append(
        {
            "Category": cat,
            "MV_in (a)": a,
            "MV_not (b)": b,
            "SGAH_in (c)": c,
            "SGAH_not (d)": d,
            "MV % in-category": a / MV_total,
            "SGAH % in-category": c / SGAH_total,
            "Odds Ratio (MV/SGAH)": OR,
            "p-value (two-sided)": p,
        }
    )

res_df = pd.DataFrame(results)

# Optional multiple-comparison control (BH/FDR)
apply_bh = st.checkbox("Apply Benjamini–Hochberg FDR (q=0.05)", value=False)
if apply_bh:
    pvals = res_df["p-value (two-sided)"].values
    m = len(pvals)
    order = pvals.argsort()
    ranks = order.argsort() + 1
    bh_thresh = 0.05 * ranks / m
    res_df["BH threshold (q=0.05)"] = bh_thresh
    res_df["Significant (BH)"] = pvals <= bh_thresh

# Pretty formatting for display
show_df = res_df.copy()
show_df["MV % in-category"] = (show_df["MV % in-category"] * 100).map("{:.1f}%".format)
show_df["SGAH % in-category"] = (show_df["SGAH % in-category"] * 100).map("{:.1f}%".format)
show_df["Odds Ratio (MV/SGAH)"] = show_df["Odds Ratio (MV/SGAH)"].map(lambda x: f"{x:.3f}")
show_df["p-value (two-sided)"] = show_df["p-value (two-sided)"].map(lambda x: f"{x:.4g}")

st.subheader("Results")
st.dataframe(show_df, use_container_width=True)

# Show contingency tables
with st.expander("Show 2×2 contingency tables for each category"):
    for _, r in res_df.iterrows():
        st.markdown(
            f"""
**{r['Category']}**

|                | In Category | Not in Category |
|----------------|-------------|-----------------|
| **MV**         | {int(r['MV_in (a)'])} | {int(r['MV_not (b)'])} |
| **SGAH**       | {int(r['SGAH_in (c)'])} | {int(r['SGAH_not (d)'])} |

Odds Ratio = {float(r['Odds Ratio (MV/SGAH)']):.3f}  
p-value (two-sided) = {float(r['p-value (two-sided)']):.4g}
"""
        )
