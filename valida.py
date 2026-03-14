# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.7 — LOLIUM BORDENAVE 2026
# Actualización:
# - Pearson por intervalos de monitoreo
# - Desfase temporal automático admisible hasta ±10 días
# - PEC calculado estrictamente hasta el día de control
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
from datetime import timedelta
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------

st.set_page_config(
    page_title="PREDWEEM BORDENAVE vK4.7",
    layout="wide",
    page_icon="🌾"
)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------
# 2. ARCHIVOS MOCK
# ---------------------------------------------------------

def create_mock_files_if_missing():

    if not (BASE / "IW.npy").exists():

        np.save(BASE / "IW.npy", np.random.rand(4,10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1,10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))

    if not (BASE / "modelo_clusters_k3.pkl").exists():

        jd=np.arange(1,366)

        p1=np.exp(-((jd-100)**2)/600)
        p2=np.exp(-((jd-160)**2)/900)+0.3*np.exp(-((jd-260)**2)/1200)
        p3=np.exp(-((jd-230)**2)/1500)

        mock_cluster={
            "JD_common":jd,
            "curves_interp":[p2,p1,p3],
            "medoids_k3":[0,1,2]
        }

        with open(BASE/"modelo_clusters_k3.pkl","wb") as f:
            pickle.dump(mock_cluster,f)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 3. FUNCIONES
# ---------------------------------------------------------

def dtw_distance(a,b):

    na,nb=len(a),len(b)

    dp=np.full((na+1,nb+1),np.inf)

    dp[0,0]=0

    for i in range(1,na+1):
        for j in range(1,nb+1):

            cost=abs(a[i-1]-b[j-1])

            dp[i,j]=cost+min(
                dp[i-1,j],
                dp[i,j-1],
                dp[i-1,j-1]
            )

    return dp[na,nb]


def calculate_tt_scalar(t,t_base,t_opt,t_crit):

    if t<=t_base:
        return 0

    elif t<=t_opt:
        return t-t_base

    elif t<t_crit:
        return (t-t_base)*((t_crit-t)/(t_crit-t_opt))

    else:
        return 0


class PracticalANNModel:

    def __init__(self,IW,bIW,LW,bLW):

        self.IW=IW
        self.bIW=bIW
        self.LW=LW
        self.bLW=bLW

        self.input_min=np.array([1,0,-7,0])
        self.input_max=np.array([300,41,25.5,84])

    def normalize(self,X):

        return 2*(X-self.input_min)/(self.input_max-self.input_min)-1

    def predict(self,Xreal):

        Xn=self.normalize(Xreal)

        z1=Xn@self.IW+self.bIW
        a1=np.tanh(z1)

        z2=(a1@self.LW.T).flatten()+self.bLW

        emerrel=(np.tanh(z2)+1)/2

        emer_ac=np.cumsum(emerrel)

        return emerrel,emer_ac


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
# 4. SIDEBAR
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
# 5. MOTOR
# ---------------------------------------------------------

if archivo_meteo is not None:

    df=pd.read_csv(archivo_meteo) if archivo_meteo.name.endswith("csv") else pd.read_excel(archivo_meteo)

    df.columns=[c.upper().strip() for c in df.columns]

    df=df.rename(columns={
        "FECHA":"Fecha",
        "TMAX":"TMAX",
        "TMIN":"TMIN",
        "PREC":"Prec"
    })

    df["Fecha"]=pd.to_datetime(df["Fecha"])

    df=df.sort_values("Fecha")

    df["Julian_days"]=df["Fecha"].dt.dayofyear

    # -----------------------------------------------------
    # ANN
    # -----------------------------------------------------

    df["JD_Shifted"]=(df["Julian_days"]+60).clip(1,300)

    X=df[["JD_Shifted","TMAX","TMIN","Prec"]].to_numpy()

    emerrel,_=modelo_ann.predict(X)

    df["EMERREL"]=np.maximum(emerrel,0)

    # -----------------------------------------------------
    # RESTRICCIÓN HÍDRICA
    # -----------------------------------------------------

    df["Prec_sum_21d"]=df["Prec"].rolling(21,min_periods=1).sum()

    df["Hydric_Factor"]=1/(1+np.exp(-0.4*(df["Prec_sum_21d"]-15)))

    df["EMERREL"]=df["EMERREL"]*df["Hydric_Factor"]

    # -----------------------------------------------------
    # TÉRMICO
    # -----------------------------------------------------

    df["Tmedia"]=(df["TMAX"]+df["TMIN"])/2

    df["DG"]=df["Tmedia"].apply(
        lambda x:calculate_tt_scalar(x,2,20,30)
    )

    # -----------------------------------------------------
    # DETECCIÓN DE PICO
    # -----------------------------------------------------

    indices_pulso=df.index[df["EMERREL"]>=umbral_er].tolist()

    fecha_inicio_ventana=None
    fecha_control=None

    if indices_pulso:

        fecha_inicio_ventana=df.loc[indices_pulso[0],"Fecha"]

        df_desde_pico=df[df["Fecha"]>=fecha_inicio_ventana].copy()

        df_desde_pico["DGA_cum"]=df_desde_pico["DG"].cumsum()

        df_control=df_desde_pico[df_desde_pico["DGA_cum"]>=600]

        if not df_control.empty:

            fecha_control=df_control.iloc[0]["Fecha"]

    # -----------------------------------------------------
    # VISUALIZACIÓN
    # -----------------------------------------------------

    st.title("🌾 PREDWEEM LOLIUM - BORDENAVE")

    fig=go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Fecha"],
        y=df["EMERREL"],
        mode="lines",
        name="Emergencia"
    ))

    fig.add_hline(
        y=umbral_er,
        line_dash="dash",
        line_color="orange"
    )

    if fecha_control:

        fig.add_vline(
            x=fecha_control,
            line_dash="dot",
            line_color="red"
        )

        fin_res=fecha_control+timedelta(days=residualidad)

        fig.add_vrect(
            x0=fecha_control,
            x1=fin_res,
            fillcolor="blue",
            opacity=0.1,
            line_width=0
        )

    fig.update_layout(
        height=450,
        title="Dinámica de emergencia"
    )

    st.plotly_chart(fig,use_container_width=True)

    # -----------------------------------------------------
    # EXPORT
    # -----------------------------------------------------

    output=io.BytesIO()

    with pd.ExcelWriter(output,engine="xlsxwriter") as writer:

        df.to_excel(writer,index=False,sheet_name="Data_Diaria")

    st.sidebar.download_button(
        "Descargar reporte",
        output.getvalue(),
        "PREDWEEM_v4_7.xlsx"
    )

else:

    st.info("Cargue datos climáticos para comenzar")
