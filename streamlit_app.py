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

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.time_series_forecasting_pipeline import TimeSeriesForecastingPipeline
from tsfm_public.toolkit.visualization import plot_predictions

from streamlit_autorefresh import st_autorefresh

# ======================================
# Auto-refresco cada 5 minutos
# ======================================
count = st_autorefresh(interval=5 * 60 * 1000, limit=None, key="refresh")

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

# Botón para refrescar manualmente la query
if st.button("Refrescar Query"):
    st.experimental_rerun()
    st.stop()

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

st.write(f"### Gráfico de {selected_param} para el equipo  {selected_equipment}")
fig, ax = plt.subplots(figsize=(12, 6))
if not filtered_data.empty:
    ax.plot(
        filtered_data['ReadTime'],
        filtered_data['ParameterFloatValue'],
        label=selected_param,
        color='blue',
        linewidth=1
    )
    ax.set_title(f"{selected_param} para  {selected_equipment}")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Valor")
    ax.legend()
    plt.grid()
    st.pyplot(fig)
else:
    st.write("No hay datos disponibles para ese parámetro.")

st.write("### Valores únicos en ParameterName")
unique_parameters = data['ParameterName'].unique()
st.write(f"Número de parámetros únicos: {len(unique_parameters)}")
st.write(unique_parameters)

data['ReadTime'] = pd.to_datetime(data['ReadTime'])
pivoted_data = data.pivot_table(
    index='ReadTime',
    columns='ParameterName',
    values='ParameterFloatValue'
)

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

display_config_file(CONFIG_PATH)

model = load_model()
observable_scaler, target_scaler = load_scalers()

resampled_data = resampled_data.reset_index()
if 'ReadTime' in resampled_data.columns:
    resampled_data.rename(columns={'ReadTime': 'New_Date/Time'}, inplace=True)

st.write("### Inspección de columnas y formatos")
st.write("#### Nombres de las columnas")
st.write(resampled_data.columns.tolist())
st.write("#### Tipos de datos de las columnas")
st.write(resampled_data.dtypes)
st.write("#### Primeros registros del DataFrame")
st.dataframe(resampled_data.head())
st.write("#### Resumen estadístico de las columnas numéricas")
st.dataframe(resampled_data.describe())
st.write("#### Verificación de valores nulos")
st.write(resampled_data.isnull().sum())

timestamp_column = "New_Date/Time"
target_column = "Engine Oil Temperature-Engine (Deg F)"
observable_columns = [col for col in resampled_data.columns if col != timestamp_column]

if model is not None and observable_scaler is not None and target_scaler is not None:
    if target_column not in resampled_data.columns or timestamp_column not in resampled_data.columns:
        st.error("Faltan columnas obligatorias en el DataFrame.")
    else:
        try:
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
                observable_columns=observable_columns,
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

            # Renombrar la columna de predicciones para distinguirla
            if target_column in predictions.columns:
                predictions.rename(columns={target_column: f"{target_column}_pred"}, inplace=True)

            # Guardar las predicciones actuales en session_state
            current_predictions = predictions.copy()
            st.session_state["previous_predictions"] = current_predictions

            # Intentar comparar predicciones previas con valores reales actuales
            merged = pd.merge(st.session_state["previous_predictions"], resampled_data, on=timestamp_column, how="inner")
            if target_column in merged.columns and f"{target_column}_pred" in merged.columns:
                merged["error"] = merged[f"{target_column}_pred"] - merged[target_column]
                mae = mean_absolute_error(merged[target_column], merged[f"{target_column}_pred"])
                rmse = np.sqrt(mean_squared_error(merged[target_column], merged[f"{target_column}_pred"]))

                st.write("### Validación Continua")
                st.write(f"MAE: {mae}")
                st.write(f"RMSE: {rmse}")

                # Graficar bullseye chart del error
                fig_bullseye, ax_bullseye = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6,6))
                angles = np.linspace(0, 2*np.pi, len(merged), endpoint=False)
                r = np.abs(merged["error"].values)
                sc = ax_bullseye.scatter(angles, r, c=r, cmap='RdYlGn_r', s=50)
                ax_bullseye.set_yticklabels([])
                ax_bullseye.set_xticklabels([])
                ax_bullseye.set_title("Bullseye Chart del Error (más cerca del centro = mejor)", y=1.08)
                ax_bullseye.scatter([0], [0], c='black', marker='x', s=100)
                st.pyplot(fig_bullseye)
            else:
                st.write("Aún no hay valores reales para las marcas de tiempo predichas. Espera a que la consulta se refresque y lleguen datos reales.")

        except Exception as e:
            st.error(f"Error durante la predicción: {e}")
else:
    st.error("No se pueden realizar predicciones porque el modelo o los escaladores no están cargados correctamente.")
