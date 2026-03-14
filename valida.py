
# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.8 — LOLIUM BORDENAVE 2026
# Nueva funcionalidad:
# - Detección automática de cohortes de emergencia
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
from datetime import timedelta
from pathlib import Path

st.set_page_config(
    page_title="PREDWEEM BORDENAVE vK4.8",
    layout="wide",
    page_icon="🌾"
)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# MOCK FILES
# ---------------------------------------------------------

def create_mock_files_if_missing():

    if not (BASE / "IW.npy").exists():

        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))

    if not (BASE / "modelo_clusters_k3.pkl").exists():

        jd = np.arange(1, 366)

        p1 = np.exp(-((jd - 100)**2)/600)
        p2 = np.exp(-((jd - 160)**2)/900) + 0.3*np.exp(-((jd - 260)**2)/1200)
        p3 = np.exp(-((jd - 230)**2)/1500)

        mock_cluster = {
            "JD_common": jd,
            "curves_interp": [p2, p1, p3],
            "medoids_k3": [0, 1, 2]
        }

        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

create_mock_files_if_missing()

# ---------------------------------------------------------
# ANN MODEL
# ---------------------------------------------------------

class PracticalANNModel:

    def __init__(self, IW, bIW, LW, bLW):

        self.IW = IW
        self.bIW = bIW
        self.LW = LW
        self.bLW = bLW

        self.input_min = np.array([1,0,-7,0])
        self.input_max = np.array([300,41,25.5,84])

    def normalize(self,X):

        return 2*(X-self.input_min)/(self.input_max-self.input_min)-1

    def predict(self,Xreal):

        Xn = self.normalize(Xreal)

        z1 = Xn @ self.IW + self.bIW
        a1 = np.tanh(z1)

        z2 = (a1 @ self.LW.T).flatten() + self.bLW

        emerrel = (np.tanh(z2)+1)/2

        emer_ac = np.cumsum(emerrel)

        return emerrel, emer_ac

# ---------------------------------------------------------
# DETECCION DE COHORTES
# ---------------------------------------------------------

def detect_emergence_cohorts(df, threshold=0.05, min_gap_days=7):

    cohorts=[]
    in_cohort=False
    start_date=None
    gap_counter=0

    for i,row in df.iterrows():

        if row["EMERREL"]>=threshold:

            if not in_cohort:
                start_date=row["Fecha"]
                in_cohort=True

            gap_counter=0

        else:

            if in_cohort:
                gap_counter+=1

                if gap_counter>=min_gap_days:

                    end_date=df.loc[i-gap_counter,"Fecha"]

                    cohorts.append({
                        "inicio":start_date,
                        "fin":end_date
                    })

                    in_cohort=False
                    gap_counter=0

    if in_cohort:

        cohorts.append({
            "inicio":start_date,
            "fin":df["Fecha"].iloc[-1]
        })

    return cohorts

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

@st.cache_resource
def load_models():

    ann=PracticalANNModel(
        np.load(BASE/"IW.npy"),
        np.load(BASE/"bias_IW.npy"),
        np.load(BASE/"LW.npy"),
        np.load(BASE/"bias_out.npy")
    )

    with open(BASE/"modelo_clusters_k3.pkl","rb") as f:
        k3=pickle.load(f)

    return ann,k3

modelo_ann,cluster_model=load_models()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("Datos")

archivo_meteo=st.sidebar.file_uploader(
    "Clima",
    type=["xlsx","csv"]
)

archivo_campo=st.sidebar.file_uploader(
    "Campo",
    type=["xlsx","csv"]
)

umbral_er=st.sidebar.slider(
    "Umbral alerta",
    0.05,
    0.8,
    0.15
)

residualidad=st.sidebar.number_input(
    "Residualidad herbicida",
    0,
    60,
    20
)

# ---------------------------------------------------------
# CARGA METEO
# ---------------------------------------------------------

if archivo_meteo is not None:

    df=pd.read_csv(archivo_meteo) if archivo_meteo.name.endswith("csv") else pd.read_excel(archivo_meteo)

    df.columns=[c.upper() for c in df.columns]

    df=df.rename(columns={
        "FECHA":"Fecha",
        "TMAX":"TMAX",
        "TMIN":"TMIN",
        "PREC":"Prec"
    })

    df["Fecha"]=pd.to_datetime(df["Fecha"])

    df=df.sort_values("Fecha")

    df["Julian_days"]=df["Fecha"].dt.dayofyear

    # ANN PREDICCION

    df["JD_Shifted"]=(df["Julian_days"]+60).clip(1,300)

    X=df[["JD_Shifted","TMAX","TMIN","Prec"]].to_numpy()

    emerrel,_=modelo_ann.predict(X)

    df["EMERREL"]=np.maximum(emerrel,0)

    # Restriccion hidrica

    df["Prec_sum_21d"]=df["Prec"].rolling(21,min_periods=1).sum()

    df["Hydric_Factor"]=1/(1+np.exp(-0.4*(df["Prec_sum_21d"]-15)))

    df["EMERREL"]=df["EMERREL"]*df["Hydric_Factor"]

    # ---------------------------------------------------------
    # DETECCION DE COHORTES
    # ---------------------------------------------------------

    cohortes=detect_emergence_cohorts(
        df,
        threshold=umbral_er*0.5,
        min_gap_days=7
    )

    for c in cohortes:

        mask=(df["Fecha"]>=c["inicio"])&(df["Fecha"]<=c["fin"])

        c["emerg_total"]=df.loc[mask,"EMERREL"].sum()

        pico_idx=df.loc[mask,"EMERREL"].idxmax()

        c["pico"]=df.loc[pico_idx,"Fecha"]
        c["pico_valor"]=df.loc[pico_idx,"EMERREL"]

    # ---------------------------------------------------------
    # GRAFICO
    # ---------------------------------------------------------

    st.title("🌾 PREDWEEM LOLIUM")

    fig=go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Fecha"],
        y=df["EMERREL"],
        mode="lines",
        fill="tozeroy",
        name="Emergencia diaria"
    ))

    fig.add_hline(
        y=umbral_er,
        line_dash="dash",
        line_color="orange"
    )

    # Cohortes

    for c in cohortes:

        fig.add_vrect(
            x0=c["inicio"],
            x1=c["fin"],
            fillcolor="purple",
            opacity=0.08,
            line_width=0
        )

        fig.add_vline(
            x=c["pico"],
            line_dash="dot",
            line_color="purple"
        )

    fig.update_layout(
        height=450,
        title="Dinámica de emergencia con cohortes"
    )

    st.plotly_chart(fig,use_container_width=True)

    # ---------------------------------------------------------
    # METRICAS
    # ---------------------------------------------------------

    col1,col2,col3=st.columns(3)

    col1.metric(
        "Cohortes detectadas",
        len(cohortes)
    )

    if cohortes:

        cohorte_principal=max(
            cohortes,
            key=lambda x:x["emerg_total"]
        )

        col2.metric(
            "Pico cohorte principal",
            cohorte_principal["pico"].strftime("%d-%m")
        )

        col3.metric(
            "Emergencia cohorte principal",
            f"{cohorte_principal['emerg_total']:.2f}"
        )

    # ---------------------------------------------------------
    # TABLA
    # ---------------------------------------------------------

    if cohortes:

        tabla=pd.DataFrame(cohortes)

        st.subheader("Cohortes detectadas")

        st.dataframe(tabla)

    # ---------------------------------------------------------
    # EXPORT
    # ---------------------------------------------------------

    output=io.BytesIO()

    with pd.ExcelWriter(output,engine="xlsxwriter") as writer:

        df.to_excel(writer,index=False,sheet_name="Emergencia")

        if cohortes:
            pd.DataFrame(cohortes).to_excel(writer,index=False,sheet_name="Cohortes")

    st.sidebar.download_button(
        "Descargar reporte",
        output.getvalue(),
        "PREDWEEM_cohortes.xlsx"
    )
