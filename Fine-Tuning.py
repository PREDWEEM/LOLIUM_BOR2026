import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io

# 1. CARGA DE PESOS ORIGINALES (SEMILLA)
IW = np.load('IW.npy')
bIW = np.load('bias_IW.npy')
LW = np.load('LW.npy')
bLW = np.load('bias_out.npy')

# 2. CARGA DE DATOS DEL SITIO 2
df_m = pd.read_csv('meteo_daily (2).csv')
df_m['Fecha'] = pd.to_datetime(df_m['Fecha'])
df_m['Julian_days'] = df_m['Fecha'].dt.dayofyear
df_m['Prec_sum'] = df_m['Prec'].rolling(window=21, min_periods=1).sum()

df_c = pd.read_csv('VALIDA (1).xlsx - Hoja1.csv')
df_c['FECHA'] = pd.to_datetime(df_c['FECHA'])
df_c['ER_obs'] = df_c['PLM2'] / df_c['PLM2'].max()

# 3. MOTOR DE LA RED NEURONAL
def predict_ann(params, X_raw):
    # Reconstrucción de los 331 parámetros
    ptr = 0
    new_IW = params[ptr:ptr+220].reshape(4, 55); ptr += 220
    new_bIW = params[ptr:ptr+55]; ptr += 55
    new_LW = params[ptr:ptr+55].reshape(1, 55); ptr += 55
    new_bLW = params[ptr]

    # Escalamiento original del modelo PREDWEEM
    in_min = np.array([1, 0, -7, 0])
    in_max = np.array([300, 41, 25.5, 84])
    X_n = 2 * (X_raw - in_min) / (in_max - in_min) - 1
    
    emer = []
    for x in X_n:
        a1 = np.tanh(new_IW.T @ x + new_bIW)
        z2 = np.dot(new_LW, a1) + new_bLW
        emer.append((np.tanh(z2) + 1) / 2)
    return np.diff(np.cumsum(np.array(emer).flatten()), prepend=0)

X_all = df_m[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)

# 4. OPTIMIZACIÓN
def objective(params):
    preds = predict_ann(params, X_all)
    # Flexibilizamos filtros: menos restrictivos para el sitio 2
    preds[(df_m['Prec_sum'] < 12) | (df_m['Julian_days'] <= 10)] = 0.0
    
    y_p_adj = []
    for f_o in df_c['FECHA']:
        mask = (df_m['Fecha'] >= f_o - pd.Timedelta(days=3)) & (df_m['Fecha'] <= f_o + pd.Timedelta(days=3))
        y_p_adj.append(preds[mask].max() if any(mask) else 0)
    
    return np.mean((df_c['ER_obs'].values - np.array(y_p_adj))**2)

# Ejecutar optimización (Ajuste global)
x0 = np.concatenate([IW.flatten(), bIW, LW.flatten(), [bLW]])
res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100})

# 5. RESULTADOS Y GUARDADO
params_opt = res.x
np.save('IW_SITE2.npy', params_opt[0:220].reshape(4, 55))
np.save('bias_IW_SITE2.npy', params_opt[220:275])
np.save('LW_SITE2.npy', params_opt[275:330].reshape(1, 55))
np.save('bias_out_SITE2.npy', params_opt[330])

print(f"✅ Calibración completada para el Sitio 2.")
print(f"Error final: {res.fun:.4f}")
