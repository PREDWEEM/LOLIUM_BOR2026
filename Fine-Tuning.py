import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path

# 1. CARGA DE COMPONENTES ORIGINALES
IW = np.load('IW.npy')
bIW = np.load('bias_IW.npy')
LW_init = np.load('LW.npy')
bLW_init = np.load('bias_out.npy')

# 2. PREPARACIÓN DE DATOS DEL SITIO 2
df_m = pd.read_csv('meteo_daily (2).csv')
df_m['Fecha'] = pd.to_datetime(df_m['Fecha'])
df_m['Julian_days'] = df_m['Fecha'].dt.dayofyear
df_m['Prec_sum'] = df_m['Prec'].rolling(window=21, min_periods=1).sum()

df_c = pd.read_csv('VALIDA (1).xlsx - Hoja1.csv')
df_c['FECHA'] = pd.to_datetime(df_c['FECHA'])
df_c['ER_obs'] = df_c['PLM2'] / df_c['PLM2'].max()

# 3. FUNCIÓN DE PREDICCIÓN CON PESOS VARIABLES
def predict_with_params(params, X_input):
    # params[0:55] = LW (1x55), params[55] = bias_out
    new_LW = params[:55].reshape(1, 55)
    new_bLW = params[55]
    
    # Normalización (Hardcoded del modelo original)
    in_min = np.array([1, 0, -7, 0])
    in_max = np.array([300, 41, 25.5, 84])
    X_n = 2 * (X_input - in_min) / (in_max - in_min) - 1
    
    emer_rel = []
    for x in X_n:
        # Capa oculta fija (Preservamos la inteligencia climática)
        a1 = np.tanh(IW.T @ x + bIW)
        # Capa de salida optimizable
        z2 = np.dot(new_LW, a1) + new_bLW
        emer_rel.append((np.tanh(z2) + 1) / 2)
    
    return np.diff(np.cumsum(np.array(emer_rel).flatten()), prepend=0)

# 4. FUNCIÓN OBJETIVO (Minimizar 1 - NSE)
def objective(params):
    # Generar predicción diaria
    X_all = df_m[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    preds_daily = predict_with_params(params, X_input=X_all)
    
    # Aplicar filtros biológicos (Ajustados para el sitio 2)
    preds_daily[(df_m['Prec_sum'] < 15) | (df_m['Julian_days'] <= 10)] = 0.0
    
    # Sincronización con ventana de 7 días
    y_pred_adj = []
    for f_obs in df_c['FECHA']:
        mask = (df_m['Fecha'] >= f_obs - pd.Timedelta(days=3)) & (df_m['Fecha'] <= f_obs + pd.Timedelta(days=3))
        y_pred_adj.append(preds_daily[mask].max() if any(mask) else 0)
    
    y_o = df_c['ER_obs'].values
    y_p = np.array(y_pred_adj)
    
    # Error Cuadrático Medio
    return np.mean((y_o - y_p)**2)

# 5. EJECUCIÓN DE LA OPTIMIZACIÓN
# Vector inicial: pesos actuales aplanados
x0 = np.append(LW_init.flatten(), bLW_init)

print("Optimizando pesos para el nuevo sitio... por favor espere.")
res = minimize(objective, x0, method='BFGS', options={'disp': True, 'maxiter': 50})

# 6. GUARDAR NUEVOS PESOS
optimized_params = res.x
new_LW = optimized_params[:55].reshape(1, 55)
new_bLW = optimized_params[55]

np.save('LW_SITIO2.npy', new_LW)
np.save('bias_out_SITIO2.npy', new_bLW)

print("\n✅ ¡Optimización Completada!")
print(f"Error Final (MSE): {res.fun:.4f}")
