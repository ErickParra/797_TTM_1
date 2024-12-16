import os
import json
import pyodbc
import torch
import joblib
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from streamlit_autorefresh import st_autorefresh

# Importaciones del modelo y pipeline
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.time_series_forecasting_pipeline import TimeSeriesForecastingPipeline
from tsfm_public.toolkit.visualization import plot_predictions

# ======================================
# Auto-refresco cada 1 minuto (60 segundos)
# ======================================
count = st_autorefresh(interval=60 * 1000, limit=None, key="auto_refresh")

# ======================================
# Ajustes de paths y archivos del modelo
# ======================================
MODEL_DIR = "."
MODEL_PATH = "./model.safetensors"
OBSERVABLE_SCALER_PATH = "./observable_scaler_0.pkl"
TARGET_SCALER_PATH = "./target_scaler_0.pkl"
CONFIG_PATH = "./config.json"

if "previous_predictions" not in st.session_state:
    st.session_state["previous_predictions"] = pd.DataFrame()

@st.cache_data
def load_data(query, conn_str):
    try:
        conn = pyodbc.connect(conn_str)
        data = pd.read_sql(query, conn)
        conn.close()
        return data
    except pyodbc.Error as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return pd.DataFrame()

def display_config_file(config_path):
    try:
        with open(config_path, "r") as file:
            config_data = json.load(file)
        st.write("### Contenido del archivo config.json")
        st.json(config_data)
    except FileNotFoundError:
        st.error(f"El archivo {config_path} no se encontró.")
    except json.JSONDecodeError as e:
        st.error(f"Error al leer el archivo JSON: {e}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")

def convert_units(resampled_data):
    # Conversión de presión: kPa a psi
    pressure_columns = [
        "Engine Oil Pressure-Engine (psi)",
        "Service Brake Accumulator Pressure-Brake ECM (psi)",
        "Differential (Axle) Lube Pressure-Brake ECM (psi)",
        "Steering Accumulator Oil Pressure-Chassis Ctrl (psi)",
        "Machine System Air Pressure-Chassis Ctrl (psi)",
        "Intake Manifold #2 Pressure-Engine (psi)",
        "Intake Manifold Pressure-Engine (psi)",
        "Left Rear Parking Brake Oil Pressure-Brake ECM (psi)",
        "Fuel Pressure-Engine (psi)",
        "Right Rear Parking Brake Oil Pressure-Brake ECM (psi)",
        "Oil Filter Differential Pressure-Engine (psi)",
        "Engine Coolant Pump Outlet Pressure (absolute)-Engine (psi)",
        "Desired Fuel Rail Pressure-Engine (psi)",
        "Fuel Rail Pressure-Engine (psi)"
    ]
    for col in pressure_columns:
        if col in resampled_data.columns:
            resampled_data[col] = resampled_data[col] * 0.145038

    # Conversión de temperatura: °C a °F
    temperature_columns = [
        "Intake Manifold Air Temperature-Engine (Deg F)",
        "Intake Manifold #2 Air Temperature-Engine (Deg F)",
        "Right Exhaust Temperature-Engine (Deg F)",
        "Left Exhaust Temperature-Engine (Deg F)",
        "Left Front Brake Oil Temperature-Brake ECM (Deg F)",
        "Right Front Brake Oil Temperature-Brake ECM (Deg F)",
        "Right Rear Brake Oil Temperature-Brake ECM (Deg F)",
        "Left Rear Brake Oil Temperature-Brake ECM (Deg F)",
        "Engine Coolant Pump Outlet Temperature-Engine (Deg F)",
        "Engine Coolant Temperature-Engine (Deg F)",
        "Transmission Oil Temperature-Trans Ctrl (Deg F)",
        "Engine Oil Temperature-Engine (Deg F)"
    ]
    for col in temperature_columns:
        if col in resampled_data.columns:
            resampled_data[col] = resampled_data[col] * 9 / 5 + 32

    return resampled_data

@st.cache_resource
def load_model():
    try:
        model = TinyTimeMixerForPrediction.from_pretrained(
            pretrained_model_name_or_path=MODEL_DIR,
            from_tf=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
        st.success("Modelo TTM cargado correctamente.")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo TTM: {e}")
        return None

@st.cache_resource
def load_scalers():
    try:
        observable_scaler = joblib.load(OBSERVABLE_SCALER_PATH)
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        st.success("Escaladores cargados correctamente.")
        return observable_scaler, target_scaler
    except Exception as e:
        st.error(f"Error al cargar los escaladores: {e}")
        return None, None

server = st.secrets["server"]
database = st.secrets["database"]
username = st.secrets["username"]
password = st.secrets["password"]
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

if "selected_equipment" not in st.session_state:
    st.session_state["selected_equipment"] = "C17"

available_equipments = ['C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C32', 'C34', 'C36', 'C37', 'C38', 'C39', 'C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46', 'C47', 'C48']

st.write("### Selección de Equipo Inicial")
selected_equipment = st.selectbox(
    "Seleccione el equipo (inicial):",
    available_equipments,
    index=available_equipments.index(st.session_state["selected_equipment"]),
    key="initial_selector"
)

if st.session_state["selected_equipment"] != selected_equipment:
    st.session_state["selected_equipment"] = selected_equipment

# Botón para refrescar manualmente la query
if st.button("Refrescar Datos Manualmente"):
    st.experimental_rerun()
    st.stop()

query = f"""
SELECT
       [EquipmentName],
       [ReadTime],
       [EquipmentModel],
       [ParameterName],
       [ParameterFloatValue]
FROM [OemDataProvider].[OemParameterExternalView]
WHERE 
      [EquipmentModel] = '797F' AND 
      [EquipmentName] = '{selected_equipment}' AND 
      [ParameterName] IN (
          'Parking Brake (797F)', 
          'Cold Mode (797F)', 
          'Shift Lever Position (797F)',
          'Oht Truck Payload State (797F)', 
          'Engine Oil Pressure (797F)', 
          'Service Brake Accumulator Pressure (797F)',
          'Differential (Axle) Lube Pressure (797F)', 
          'Steering Accumulator Oil Pressure (797F)',
          'Intake Manifold Air Temperature (797F)', 
          'Intake Manifold #2 Air Temperature (797F)', 
          'Machine System Air Pressure (797F)',
          'Intake Manifold #2 Pressure (797F)', 
          'Intake Manifold Pressure (797F)', 
          'Left Rear Parking Brake Oil Pressure (797F)',
          'Fuel Pressure (797F)', 
          'Transmission Input Speed (797F)', 
          'Engine Coolant Pump Outlet Pressure (797F)', 
          'Engine Speed (797F)',
          'Fuel Rail Pressure (797F)', 
          'Engine Fan Speed (797F)',
          'Right Exhaust Temperature (797F)', 
          'Left Exhaust Temperature (797F)', 
          'Left Front Brake Oil Temperature (797F)',
          'Right Front Brake Oil Temperature (797F)', 
          'Oil Filter Differential Pressure (797F)', 
          'Right Rear Brake Oil Temperature (797F)',
          'Left Rear Brake Oil Temperature (797F)', 
          'Engine Coolant Pump Outlet Temperature (797F)', 
          'Engine Coolant Temperature (797F)',
          'Transmission Oil Temperature (797F)', 
          'Engine Oil Temperature (797F)'
      ) AND
      ParameterFloatValue IS NOT NULL AND 
      ReadTime > (DATEADD(HOUR, -120, GETDATE()))
"""

with st.spinner('Ejecutando consulta...'):
    data = load_data(query, conn_str)
st.success('Consulta completada!')

if data.empty:
    st.error("No se encontraron datos para el equipo seleccionado.")
    st.stop()

st.write(f"### Datos obtenidos para el equipo: {selected_equipment}")
st.dataframe(data)

selected_param = st.selectbox(
    "Seleccione un parámetro para graficar:",
    data['ParameterName'].unique()
)

data.loc[data['ParameterFloatValue'] == 32784, 'ParameterFloatValue'] = 0
filtered_data = data[(data['ParameterName'] == selected_param) & (data['ParameterFloatValue'] >= -100)]
filtered_data['ReadTime'] = pd.to_datetime(filtered_data['ReadTime'])
filtered_data = filtered_data.sort_values(by='ReadTime')

st.write(f"### Gráfico de {selected_param} para el equipo {selected_equipment}")
fig, ax = plt.subplots(figsize=(12, 6))
if not filtered_data.empty:
    ax.plot(
        filtered_data['ReadTime'],
        filtered_data['ParameterFloatValue'],
        label=selected_param,
        color='blue',
        linewidth=1
    )
    ax.set_title(f"{selected_param} para {selected_equipment}")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Valor")
    ax.legend()
    plt.grid()
    st.pyplot(fig)
else:
    st.write("No hay datos disponibles para ese parámetro.")

# Proceso de resample, conversión de unidades, etc. (asumimos ya hecho arriba)
# ...


resampled_data = pivoted_data.resample('30S').mean().interpolate(method='linear')

st.write("### Datos resampleados a 30 segundos")
st.dataframe(resampled_data.head())

vims_column_mapping = {
    "Parking Brake (797F)": "Parking Brake-Brake ECM ()",
    "Cold Mode (797F)": "Cold Mode-Engine ()",
    "Shift Lever Position (797F)": "Shift Lever Position-Chassis Ctrl ()",
    "Oht Truck Payload State (797F)": "OHT Truck Payload State-Communication Gateway #2 ()",
    "Engine Oil Pressure (797F)": "Engine Oil Pressure-Engine (psi)",
    "Service Brake Accumulator Pressure (797F)": "Service Brake Accumulator Pressure-Brake ECM (psi)",
    "Differential (Axle) Lube Pressure (797F)": "Differential (Axle) Lube Pressure-Brake ECM (psi)",
    "Steering Accumulator Oil Pressure (797F)": "Steering Accumulator Oil Pressure-Chassis Ctrl (psi)",
    "Intake Manifold Air Temperature (797F)": "Intake Manifold Air Temperature-Engine (Deg F)",
    "Intake Manifold #2 Air Temperature (797F)": "Intake Manifold #2 Air Temperature-Engine (Deg F)",
    "Machine System Air Pressure (797F)": "Machine System Air Pressure-Chassis Ctrl (psi)",
    "Intake Manifold #2 Pressure (797F)": "Intake Manifold #2 Pressure-Engine (psi)",
    "Intake Manifold Pressure (797F)": "Intake Manifold Pressure-Engine (psi)",
    "Left Rear Parking Brake Oil Pressure (797F)": "Left Rear Parking Brake Oil Pressure-Brake ECM (psi)",
    "Fuel Pressure (797F)": "Fuel Pressure-Engine (psi)",
    "Right Rear Parking Brake Oil Pressure (797F)": "Right Rear Parking Brake Oil Pressure-Brake ECM (psi)",
    "Transmission Input Speed (797F)": "Transmission Input Speed-Trans Ctrl (rpm)",
    "Engine Coolant Pump Outlet Pressure (797F)": "Engine Coolant Pump Outlet Pressure (absolute)-Engine (psi)",
    "Engine Speed (797F)": "Engine Speed-Engine (rpm)",
    "Desired Fuel Rail Pressure (797F)": "Desired Fuel Rail Pressure-Engine (psi)",
    "Fuel Rail Pressure (797F)": "Fuel Rail Pressure-Engine (psi)",
    "Engine Fan Speed (797F)": "Engine Fan Speed-Brake ECM (rpm)",
    "Right Exhaust Temperature (797F)": "Right Exhaust Temperature-Engine (Deg F)",
    "Left Exhaust Temperature (797F)": "Left Exhaust Temperature-Engine (Deg F)",
    "Left Front Brake Oil Temperature (797F)": "Left Front Brake Oil Temperature-Brake ECM (Deg F)",
    "Right Front Brake Oil Temperature (797F)": "Right Front Brake Oil Temperature-Brake ECM (Deg F)",
    "Oil Filter Differential Pressure (797F)": "Oil Filter Differential Pressure-Engine (psi)",
    "Right Rear Brake Oil Temperature (797F)": "Right Rear Brake Oil Temperature-Brake ECM (Deg F)",
    "Left Rear Brake Oil Temperature (797F)": "Left Rear Brake Oil Temperature-Brake ECM (Deg F)",
    "Engine Coolant Pump Outlet Temperature (797F)": "Engine Coolant Pump Outlet Temperature-Engine (Deg F)",
    "Engine Coolant Temperature (797F)": "Engine Coolant Temperature-Engine (Deg F)",
    "Transmission Oil Temperature (797F)": "Transmission Oil Temperature-Trans Ctrl (Deg F)",
    "Engine Oil Temperature (797F)": "Engine Oil Temperature-Engine (Deg F)"
}

resampled_data.rename(columns=vims_column_mapping, inplace=True)
resampled_data = convert_units(resampled_data)

resampled_data.index = resampled_data.index.tz_localize('UTC').tz_convert('America/Santiago')
resampled_data = resampled_data.sort_index(ascending=False)

st.write("### Datos procesados después de conversiones y renombrados")
st.dataframe(resampled_data.head())



# target_column y timestamp_column establecidos
timestamp_column = "New_Date/Time"
target_column = "Engine Oil Temperature-Engine (Deg F)"

model = load_model()
observable_scaler, target_scaler = load_scalers()

st.write("### Simulación de predicciones (Opcional)")
uploaded_file = st.file_uploader("Sube un archivo CSV con datos de entrada", type="csv")
if uploaded_file and observable_scaler is not None and target_scaler is not None and model is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Datos de entrada cargados:")
    st.dataframe(input_data)
    try:
        normalized_data = observable_scaler.transform(input_data.values)
        input_tensor = torch.tensor(normalized_data).unsqueeze(0)
        with torch.no_grad():
            raw_predictions = model(input_tensor)
        descaled_predictions = target_scaler.inverse_transform(raw_predictions.numpy().squeeze(0))
        st.write("Predicciones desescaladas:")
        st.dataframe(descaled_predictions)
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")

if model is not None and observable_scaler is not None and target_scaler is not None:
    if target_column not in filtered_data.columns:
        # Asegúrate de que tengas el target_column en tus datos finales
        # Ajusta la lógica según tu pipeline.
        st.error("La columna objetivo no se encuentra en los datos. Ajusta el código.")
    else:
        try:
            # Ejecutar el pipeline y obtener `predictions`
            # Ajusta observable_columns según tu caso
            observable_columns = [c for c in filtered_data.columns if c not in ["EquipmentName","EquipmentModel","ParameterName","ParameterFloatValue","ReadTime"]]

            # Suponiendo que ya tienes `resampled_data` con el target
            # Asegúrate que `resampled_data` contenga timestamp_column y target_column
            # Ajusta aquí si es necesario
            # Filtramos los últimos 512 puntos
            resampled_data = filtered_data.rename(columns={"ReadTime": timestamp_column})
            resampled_data = resampled_data[[timestamp_column, "ParameterFloatValue"]]
            resampled_data = resampled_data.rename(columns={"ParameterFloatValue": target_column})
            resampled_data = resampled_data.sort_values(by=timestamp_column)

            context_length = 512
            if len(resampled_data) > context_length:
                resampled_data = resampled_data.iloc[-context_length:]

            freq = "30S"
            pipeline = TimeSeriesForecastingPipeline(
                model=model,
                id_columns=[],
                timestamp_column=timestamp_column,
                target_columns=[target_column],
                observable_columns=[target_column], # Ajustar según tu caso
                prediction_length=96,
                context_length=context_length,
                freq=freq,
                observable_scaler=observable_scaler,
                target_scaler=target_scaler,
            )

            predictions = pipeline(
                resampled_data,
                explode_forecasts=True,
                add_known_ground_truth=True,
                inverse_scale_outputs=True,
            )

            st.write("### Predicciones generadas")
            st.dataframe(predictions)
            st.write("### Columnas en el DataFrame de predicciones:")
            st.write(predictions.columns.tolist())

            # Supongamos que predictions tiene la columna target_column con valores reales y predichos mezclados.
            # Renombramos la columna predicha para distinguirla.
            # Aquí asumiendo que las predicciones futuras están en la misma columna. Ajusta según tu pipeline.
            # Ejemplo: renombramos el target a target_column+"_pred"
            predictions.rename(columns={target_column: f"{target_column}_pred"}, inplace=True)

            # Merge con datos reales (resampled_data) para comparar
            merged = pd.merge(predictions[[timestamp_column, f"{target_column}_pred"]],
                              resampled_data[[timestamp_column, target_column]],
                              on=timestamp_column, how="inner")

            # Calcular error
            merged["error"] = merged[f"{target_column}_pred"] - merged[target_column]
            mae = mean_absolute_error(merged[target_column], merged[f"{target_column}_pred"])
            rmse = np.sqrt(mean_squared_error(merged[target_column], merged[f"{target_column}_pred"]))

            st.write("### Comparación Real vs Predicho")
            st.dataframe(merged[[timestamp_column, target_column, f"{target_column}_pred", "error"]])
            st.write(f"MAE: {mae}, RMSE: {rmse}")

            # Gráfico de dispersión Real vs Predicho
            fig_comp, ax_comp = plt.subplots(figsize=(6,6))
            ax_comp.scatter(merged[target_column], merged[f"{target_column}_pred"], c='green', alpha=0.5)
            ax_comp.set_xlabel("Real")
            ax_comp.set_ylabel("Predicho")
            ax_comp.set_title("Real vs Predicho (Scatter)")
            ax_comp.grid(True)
            st.pyplot(fig_comp)

            # Gráfico de líneas comparativo en el tiempo
            fig_line, ax_line = plt.subplots(figsize=(12,6))
            ax_line.plot(merged[timestamp_column], merged[target_column], label="Real", color="blue")
            ax_line.plot(merged[timestamp_column], merged[f"{target_column}_pred"], label="Predicho", color="red", linestyle="--")
            ax_line.set_title("Real vs Predicho en el tiempo")
            ax_line.set_xlabel("Tiempo")
            ax_line.set_ylabel("Valor")
            ax_line.legend()
            ax_line.grid(True)
            st.pyplot(fig_line)

            # Bullseye chart del error
            fig_bullseye, ax_bullseye = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))
            angles = np.linspace(0, 2*np.pi, len(merged), endpoint=False)
            r = np.abs(merged["error"].values)
            sc = ax_bullseye.scatter(angles, r, c=r, cmap='RdYlGn_r', s=50)
            ax_bullseye.set_yticklabels([])
            ax_bullseye.set_xticklabels([])
            ax_bullseye.set_title("Bullseye Chart del Error (más cerca del centro = mejor)", y=1.08)
            ax_bullseye.scatter([0], [0], c='black', marker='x', s=100)
            st.pyplot(fig_bullseye)

            # Ventana deslizante: por ejemplo, última hora
            now = merged[timestamp_column].max()
            time_window = timedelta(minutes=60)
            window_start = now - time_window
            predictions_window = merged[merged[timestamp_column] >= window_start]

            fig_window, ax_window = plt.subplots(figsize=(12,6))
            ax_window.plot(predictions_window[timestamp_column], predictions_window[target_column], label="Real", color="blue")
            ax_window.plot(predictions_window[timestamp_column], predictions_window[f"{target_column}_pred"], label="Predicho", color="red", linestyle="--")
            ax_window.set_title("Ventana deslizante (última hora)")
            ax_window.set_xlabel("Tiempo")
            ax_window.set_ylabel("Valor")
            ax_window.legend()
            ax_window.grid(True)
            st.pyplot(fig_window)

            # Guardar predicciones actuales
            st.session_state["previous_predictions"] = predictions.copy()

        except Exception as e:
            st.error(f"Error durante la comparación real vs predicho: {e}")
else:
    st.error("No se pueden realizar predicciones porque el modelo o los escaladores no están cargados correctamente.")
