import os
import json
import pyodbc
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.time_series_forecasting_pipeline import TimeSeriesForecastingPipeline

# =========================
# Configuración de paths
# =========================
MODEL_PATH = "./model.safetensors"
OBSERVABLE_SCALER_PATH = "./observable_scaler_0.pkl"
TARGET_SCALER_PATH = "./target_scaler_0.pkl"

# Configuración de conexión a la base de datos
server = st.secrets["server"]
database = st.secrets["database"]
username = st.secrets["username"]
password = st.secrets["password"]
conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"

# Estado inicial de la sesión
if "query_data" not in st.session_state:
    st.session_state["query_data"] = pd.DataFrame()

if "real_vs_predicted" not in st.session_state:
    st.session_state["real_vs_predicted"] = pd.DataFrame()


def convert_units(data):
    # Conversión de presión: kPa a psi
    pressure_columns = ["Engine Oil Pressure (psi)", "Fuel Pressure (psi)"]
    for col in pressure_columns:
        if col in data.columns:
            data[col] = data[col] * 0.145038

    # Conversión de temperatura: °C a °F
    temperature_columns = ["Engine Coolant Temperature (Deg F)", "Engine Oil Temperature (Deg F)"]
    for col in temperature_columns:
        if col in data.columns:
            data[col] = data[col] * 9 / 5 + 32
    return data


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

@st.cache_resource
def load_model_and_scalers():
    try:
        model = TinyTimeMixerForPrediction.from_pretrained(MODEL_PATH)
        observable_scaler = joblib.load(OBSERVABLE_SCALER_PATH)
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        st.success("Modelo y escaladores cargados correctamente.")
        return model, observable_scaler, target_scaler
    except Exception as e:
        st.error(f"Error al cargar modelo o escaladores: {e}")
        return None, None, None


st.title("Aplicación de Predicción con Modelo TTM")

# Botón para refrescar la query SQL
def update_query():
    query = """
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
    data = load_data(query, conn_str)
    data['ReadTime'] = pd.to_datetime(data['ReadTime'])
    st.session_state["query_data"] = convert_units(data)

st.button("Actualizar Query SQL", on_click=update_query)

# Mostrar datos
if not st.session_state["query_data"].empty:
    st.write("### Datos de la Query Actualizados")
    st.dataframe(st.session_state["query_data"])
else:
    st.warning("Presiona el botón para cargar datos.")


model, observable_scaler, target_scaler = load_model_and_scalers()

if model and not st.session_state["query_data"].empty:
    st.write("### Generación de Predicciones")
    data = st.session_state["query_data"].copy()

    # Pivotar y preparar datos
    pivot_data = data.pivot(index="ReadTime", columns="ParameterName", values="ParameterFloatValue").reset_index()
    input_data = pivot_data.drop(columns="ReadTime").values
    normalized_data = observable_scaler.transform(input_data)

    # Predicción
    input_tensor = torch.tensor(normalized_data).unsqueeze(0)
    with torch.no_grad():
        predictions = model(input_tensor)
    descaled_predictions = target_scaler.inverse_transform(predictions.numpy().squeeze(0))

    # Comparación Real vs Predicho
    pivot_data["Predicted"] = descaled_predictions[:, 0]  # Suponiendo una predicción univariada
    st.session_state["real_vs_predicted"] = pivot_data

    # Mostrar resultados
    st.write("### Comparación Real vs Predicho")
    st.dataframe(pivot_data)

    # Gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pivot_data["ReadTime"], pivot_data["Engine Oil Temperature (Deg F)"], label="Real", color="blue")
    ax.plot(pivot_data["ReadTime"], pivot_data["Predicted"], label="Predicho", linestyle="--", color="red")
    ax.set_title("Valores Reales vs Predichos")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Temperatura (°F)")
    ax.legend()
    plt.grid()
    st.pyplot(fig)
else:
    st.warning("No se han cargado los datos o el modelo.")


if not st.session_state["real_vs_predicted"].empty:
    st.write("### Gráfico del Error en el Tiempo")
    real_vs_predicted = st.session_state["real_vs_predicted"]
    real_vs_predicted["Error"] = real_vs_predicted["Engine Oil Temperature (Deg F)"] - real_vs_predicted["Predicted"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(real_vs_predicted["ReadTime"], real_vs_predicted["Error"], color="orange", label="Error")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("Error entre Valores Reales y Predichos")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Error (°F)")
    ax.legend()
    plt.grid()
    st.pyplot(fig)
