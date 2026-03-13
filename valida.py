# -*- coding: utf-8 -*-
# ===============================================================
# 📊 PREDWEEM — SCRIPT DE VALIDACIÓN AGRONÓMICA A CAMPO
# Localidad: Bordenave | Lógica: vK4.4 (Shift +60d)
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# ---------------------------------------------------------
# 1. CLASE DEL MODELO (Arquitectura Corregida)
# ---------------------------------------------------------
class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        z1 = Xn @ self.IW + self.bIW
        a1 = np.tanh(z1)
        z2 = (a1 @ self.LW.T).flatten() + self.bLW
        # Salida matemática pura entre 0 y 1
        emerrel = (np.tanh(z2) + 1) / 2
        return emerrel

# ---------------------------------------------------------
# 2. CARGA DE DATOS Y PESOS
# ---------------------------------------------------------
print("Cargando pesos de la red neuronal...")
IW = np.load('IW.npy')
bIW = np.load('bias_IW.npy')
LW = np.load('LW.npy')
bLW = np.load('bias_out.npy')
modelo = PracticalANNModel(IW, bIW, LW, bLW)

print("Procesando datos meteorológicos y de campo...")
# Datos Climáticos
df_meteo = pd.read_csv('bordenave.csv')
df_meteo['Fecha'] = pd.to_datetime(df_meteo['Fecha'])
df_meteo = df_meteo.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
df_meteo['Julian_days'] = df_meteo['Fecha'].dt.dayofyear

# Datos de Campo
df_campo = pd.read_csv('bordenave_campo.xlsx - Hoja1.csv')
df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])
# Normalizamos el campo de 0 a 1 para poder compararlo con el modelo
max_plm2 = df_campo['PLM2'].max()
df_campo['Campo_Normalizado'] = df_campo['PLM2'] / max_plm2 if max_plm2 > 0 else 0

# ---------------------------------------------------------
# 3. EJECUCIÓN DEL MODELO (Lógica Bordenave vK4.4)
# ---------------------------------------------------------
# Desfase Temporal
df_meteo["JD_Shifted"] = (df_meteo["Julian_days"] + 60).clip(1, 300)

# Predicción ANN
X = df_meteo[["JD_Shifted", "TMAX", "TMIN", "Prec"]].to_numpy(float)
df_meteo["EMERREL"] = np.maximum(modelo.predict(X), 0.0)

# Restricción Hídrica Sigmoide
df_meteo["Prec_sum_21d"] = df_meteo["Prec"].rolling(window=21, min_periods=1).sum()
df_meteo["Hydric_Factor"] = 1 / (1 + np.exp(-0.4 * (df_meteo["Prec_sum_21d"] - 15)))
df_meteo["EMERREL"] = df_meteo["EMERREL"] * df_meteo["Hydric_Factor"]

# Relajación Dinámica
jd_thresholds = np.where(df_meteo["Prec_sum_21d"] > 50, 0, 15)
df_meteo.loc[df_meteo["Julian_days"] <= jd_thresholds, "EMERREL"] = 0.0

# ---------------------------------------------------------
# 4. CÁLCULO DE MÉTRICAS DE VALIDACIÓN
# ---------------------------------------------------------
# Cruzamos los datos exactamente en las fechas donde hubo conteos de campo
df_cruce = pd.merge(df_meteo[['Fecha', 'EMERREL']], 
                    df_campo[['FECHA', 'PLM2', 'Campo_Normalizado']], 
                    left_on='Fecha', right_on='FECHA', how='inner')

y_sim = df_cruce['EMERREL']
y_obs = df_cruce['Campo_Normalizado']

# -- A. Estadísticas Tradicionales --
pearson_r = y_sim.corr(y_obs)
rmse = np.sqrt(np.mean((y_sim - y_obs)**2))

# Índice de Willmott (d)
num = np.sum((y_sim - y_obs)**2)
den = np.sum((np.abs(y_sim - np.mean(y_obs)) + np.abs(y_obs - np.mean(y_obs)))**2)
willmott_d = 1 - (num / den)

# -- B. Métricas Agronómicas (Factibilidad a Campo) --
# 1. Peak Lag (Desfase del pico principal)
fecha_pico_modelo = df_meteo.loc[df_meteo['EMERREL'].idxmax(), 'Fecha']
fecha_pico_campo = df_campo.loc[df_campo['PLM2'].idxmax(), 'FECHA']
peak_lag_dias = (fecha_pico_modelo - fecha_pico_campo).days

# 2. PEC (Proporción de Emergencia Controlada)
# Asumimos que el productor aplica el día del pico del modelo + 3 días de margen logístico
fecha_aplicacion_teorica = fecha_pico_modelo + timedelta(days=3)
malezas_controladas = df_campo.loc[df_campo['FECHA'] <= fecha_aplicacion_teorica, 'PLM2'].sum()
malezas_totales = df_campo['PLM2'].sum()
pec = (malezas_controladas / malezas_totales) * 100

# ---------------------------------------------------------
# 5. REPORTE Y VISUALIZACIÓN
# ---------------------------------------------------------
print("\n" + "="*50)
print("🌾 REPORTE DE VALIDACIÓN: PREDWEEM EN BORDENAVE")
print("="*50)
print(f"📈 INDICADORES ESTADÍSTICOS")
print(f" - Correlación Pearson (r) : {pearson_r:.3f} (Óptimo > 0.7)")
print(f" - RMSE                    : {rmse:.3f} (Menor es mejor)")
print(f" - Índice de Willmott (d)  : {willmott_d:.3f} (Cercano a 1 es mejor)")
print("\n🚜 MÉTRICAS DE DECISIÓN A CAMPO")
print(f" - Fecha Pico Real         : {fecha_pico_campo.strftime('%d-%m-%Y')}")
print(f" - Fecha Alerta Modelo     : {fecha_pico_modelo.strftime('%d-%m-%Y')}")
print(f" - Desfase del Pico        : {peak_lag_dias} días")
print(f" - Fecha Aplicación (Pico+3): {fecha_aplicacion_teorica.strftime('%d-%m-%Y')}")
print(f" - P.E.C. (Malezas ctrl.)  : {pec:.1f}% de la cohorte total")
print("="*50 + "\n")

# Gráfico de Validación
plt.figure(figsize=(12, 6))
plt.plot(df_meteo['Fecha'], df_meteo['EMERREL'], label='Predicción Modelo (0-1)', color='#166534', linewidth=2.5)
plt.scatter(df_campo['FECHA'], df_campo['Campo_Normalizado'], color='#dc2626', s=100, label='Observado a Campo (Normalizado)', zorder=5)

# Marcar el momento de aplicación
plt.axvline(x=fecha_aplicacion_teorica, color='orange', linestyle='--', linewidth=2, label=f'Decisión de Control ({pec:.0f}% PEC)')

plt.title('Validación de Dinámica Poblacional - Lolium spp. (Bordenave 2026)', fontsize=14)
plt.ylabel('Tasa Relativa de Emergencia', fontsize=12)
plt.xlabel('Fecha', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('validacion_agronomica_bordenave.png', dpi=300)
print("Gráfico de validación guardado como 'validacion_agronomica_bordenave.png'.")
