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

from streamlit_autorefresh import st_autorefresh  # Importar el componente

# ======================================
# Ajustes de paths y archivos del modelo
# ======================================
MODEL_DIR = "."
MODEL_PATH = "./model.safetensors"
OBSERVABLE_SCALER_PATH = "./observable_scaler_0.pkl"
TARGET_SCALER_PATH = "./target_scaler_0.pkl"
CONFIG_PATH = "./config.json"

# ======================================
# Funciones auxiliares
# ======================================
@st.cache_data(show_spinner=False)
def load_data(query, conn_str, refresh):
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

# =========================
# Inicialización de estado
# =========================
if "query_data" not in st.session_state:
    st.session_state["query_data"] = pd.DataFrame()

if "real_vs_predicted" not in st.session_state:
    st.session_state["real_vs_predicted"] = pd.DataFrame()

# =========================
# Conversión de unidades
# =========================
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

# ===================================================
# Cargar Modelo y Escaladores
# ===================================================
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        # Cargar el modelo desde el directorio (debe contener config.json y model.safetensors)
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

@st.cache_resource(show_spinner=False)
def load_scalers():
    try:
        observable_scaler = joblib.load(OBSERVABLE_SCALER_PATH)
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        st.success("Escaladores cargados correctamente.")
        return observable_scaler, target_scaler
    except Exception as e:
        st.error(f"Error al cargar los escaladores: {e}")
        return None, None

# ===================================================
# Acceder a los secrets almacenados en Streamlit Cloud
# ===================================================
server = st.secrets["server"]
database = st.secrets["database"]
username = st.secrets["username"]
password = st.secrets["password"]

# ===================================================
# Configuración de la conexión a la base de datos
# ===================================================
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# ===================================================
# Inicializar el valor por defecto en session_state
# ===================================================
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

# ===================================================
# Autorefresh Configuración
# ===================================================
# Autorefresh cada 60 segundos (60000 ms), con límite de 100 refrescos
count = st_autorefresh(interval=60000, limit=100, key="autorefresh_counter")

# ===================================================
# Consulta SQL
# ===================================================
# =========================
# Cargar Datos desde SQL
# =========================

# Definir la consulta SQL una sola vez
base_query = f"""
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

# Botón para refrescar la query SQL
def update_query():
    # Generar un timestamp único para forzar la actualización de la caché
    refresh_timestamp = datetime.now().timestamp()
    new_data = load_data(base_query, conn_str, refresh=refresh_timestamp)
    if not new_data.empty:
        st.session_state["query_data"] = new_data
        st.success("Datos actualizados correctamente.")
    else:
        st.warning("La consulta no retornó datos.")

st.button("Actualizar Query SQL", on_click=update_query)

# Cargar los datos
if "query_data" in st.session_state and not st.session_state["query_data"].empty:
    data = st.session_state["query_data"]
else:
    with st.spinner('Ejecutando consulta...'):
        data = load_data(base_query, conn_str, refresh=False)
    st.session_state["query_data"] = data

st.success('Consulta completada!')

if data.empty:
    st.error("No se encontraron datos para el equipo seleccionado.")
    st.stop()

st.write(f"### Datos obtenidos para el equipo: {selected_equipment}")
st.dataframe(data)

# ==============================
# Continuación del flujo de la app
# ==============================

selected_param = st.selectbox(
    "Seleccione un parámetro para graficar:",
    data['ParameterName'].unique()
)

# Reemplazar valores específicos
data.loc[data['ParameterFloatValue'] == 32784, 'ParameterFloatValue'] = 0

# Filtrar y ordenar los datos
filtered_data = data[(data['ParameterName'] == selected_param) & (data['ParameterFloatValue'] >= -100)].copy()
filtered_data['ReadTime'] = pd.to_datetime(filtered_data['ReadTime'])
filtered_data = filtered_data.sort_values(by='ReadTime')

# Graficar el parámetro seleccionado
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

# Mostrar valores únicos en ParameterName
st.write("### Valores únicos en ParameterName")
unique_parameters = data['ParameterName'].unique()
st.write(f"Número de parámetros únicos: {len(unique_parameters)}")
st.write(unique_parameters)

# Pivotar y resamplear los datos
data['ReadTime'] = pd.to_datetime(data['ReadTime'])
pivoted_data = data.pivot_table(
    index='ReadTime',
    columns='ParameterName',
    values='ParameterFloatValue'
)

resampled_data = pivoted_data.resample('30S').mean().interpolate(method='linear')

st.write("### Datos resampleados a 30 segundos")
st.dataframe(resampled_data.head())

# Renombrar columnas según el mapeo
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

# Localizar la zona horaria y ordenar
resampled_data.index = resampled_data.index.tz_localize('UTC').tz_convert('America/Santiago')
resampled_data = resampled_data.sort_index(ascending=False)

st.write("### Datos procesados después de conversiones y renombrados")
st.dataframe(resampled_data.head())

# Mostrar archivo de configuración
display_config_file(CONFIG_PATH)

# Cargar el modelo y los escaladores
model = load_model()
observable_scaler, target_scaler = load_scalers()

# =============================
# Simulación de Predicciones (Opcional)
# =============================
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

# =============================
# Inspección de Datos
# =============================
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

# =============================
# Configuración para Predicción
# =============================
timestamp_column = "New_Date/Time"
target_column = "Engine Oil Temperature-Engine (Deg F)"
observable_columns = [col for col in resampled_data.columns if col != timestamp_column]

# =============================
# Backtesting
# =============================
def perform_backtesting(data, pipeline, context_length, freq, timestamp_column, target_column, prediction_length=96, step=96):
    """
    Realiza backtesting generando predicciones en ventanas deslizantes de datos históricos.

    Parameters:
    - data: DataFrame con las columnas de tiempo y objetivo.
    - pipeline: Pipeline de predicción.
    - context_length: Número de registros en la ventana de contexto.
    - freq: Frecuencia de los datos.
    - timestamp_column: Nombre de la columna de tiempo.
    - target_column: Nombre de la columna objetivo.
    - prediction_length: Número de pasos a predecir cada vez.
    - step: Desplazamiento de la ventana en cada iteración.

    Returns:
    - backtest_df: DataFrame con Timestamp, Predicted, Actual.
    """
    backtest_results = []
    n = len(data)
    st.write(f"Iniciando backtesting con {n} registros.")

    for i in range(context_length, n - prediction_length + 1, step):
        st.write(f"Procesando ventana {i - context_length} a {i + prediction_length}")
        train_data = data.iloc[i - context_length:i]

        try:
            forecast = pipeline(train_data)
            forecast = forecast.tail(prediction_length)

            # Obtener los datos reales correspondientes a las predicciones
            actual_data = data.iloc[i:i + prediction_length]

            # Asegurarse de que hay suficientes datos reales
            if len(actual_data) < prediction_length:
                st.warning(f"No hay suficientes datos reales para comparar en la iteración {i}.")
                continue

            # Iterar sobre las predicciones y los datos reales simultáneamente
            for pred_row, actual_row in zip(forecast.itertuples(index=False), actual_data.itertuples(index=False)):
                timestamp_pred = getattr(pred_row, timestamp_column)
                predicted = getattr(pred_row, target_column)
                timestamp_actual = getattr(actual_row, timestamp_column)
                actual = getattr(actual_row, target_column)

                # Verificar si los timestamps coinciden (puedes omitir esta verificación si estás seguro de la alineación)
                if timestamp_pred == timestamp_actual:
                    backtest_results.append({
                        'Timestamp': timestamp_pred,
                        'Predicted': predicted,
                        'Actual': actual
                    })
                else:
                    st.warning(f"Timestamps no coinciden: Predicción={timestamp_pred}, Actual={timestamp_actual}")

        except Exception as e:
            st.error(f"Error al procesar la ventana {i - context_length} a {i + prediction_length}: {e}")

    st.write(f"Backtesting finalizado. Número de resultados: {len(backtest_results)}")
    return pd.DataFrame(backtest_results)

# Solo predecir si el modelo y los escaladores están cargados
if model is not None and observable_scaler is not None and target_scaler is not None:
    if target_column not in resampled_data.columns or timestamp_column not in resampled_data.columns:
        st.error("Faltan columnas obligatorias en el DataFrame.")
    else:
        try:
            resampled_data = resampled_data.sort_values(by=timestamp_column)
            context_length = 512  # Mantener 512 como requerido por TTM
            if len(resampled_data) > context_length:
                resampled_data = resampled_data.iloc[-context_length:]
            else:
                st.warning(f"Datos insuficientes para el contexto. Se requiere al menos {context_length} registros.")

            st.write(f"Número de registros disponibles para backtesting: {len(resampled_data)}")

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

            # Gráfico Comparativo (Real vs Predicción)
            st.write("### Gráfico Comparativo: Real vs Predicción")
            # Obtener el timestamp máximo de los datos reales
            max_timestamp_real = resampled_data[timestamp_column].max()

            # Filtrar predicciones en el futuro
            df_pred = predictions[predictions[timestamp_column] > max_timestamp_real].copy()
            df_pred.rename(columns={target_column: "Predicted"}, inplace=True)

            # Obtener valores reales para los timestamps de predicción (si están disponibles)
            df_real_pred = pd.merge(df_pred, resampled_data[[timestamp_column, target_column]], 
                                    left_on=timestamp_column, right_on=timestamp_column, how='left')
            df_real_pred.rename(columns={target_column: "Actual"}, inplace=True)

            # Verificar si hay predicciones futuras
            if df_pred.empty:
                st.warning("No hay predicciones futuras disponibles para graficar.")
            else:
                fig, ax = plt.subplots(figsize=(12, 6))

                # Graficar valores reales (hasta max_timestamp_real)
                df_real = resampled_data[resampled_data[timestamp_column] <= max_timestamp_real].copy()
                ax.plot(
                    df_real[timestamp_column],
                    df_real[target_column],
                    label="Real",
                    linestyle="-",
                    color="blue",
                )

                # Graficar valores predichos
                ax.plot(
                    df_pred[timestamp_column],
                    df_pred["Predicted"],
                    label="Predicción",
                    linestyle="--",
                    color="red",
                )

                ax.set_title("Predicción vs Real")
                ax.legend()
                plt.grid()
                st.pyplot(fig)

            y_min, y_max = 150, 245

            st.write(f"### Gráfico de Predicciones (Horizonte Futuro) {selected_equipment}")
            prediction_col = "Predicted"
            if prediction_col in df_real_pred.columns and not df_pred.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(
                    df_real_pred[timestamp_column],
                    df_real_pred[prediction_col],
                    label="Predicción",
                    linestyle="--",
                    color="green",
                )
                ax.set_title(f"Predicciones Generadas (Horizonte Futuro) {selected_equipment}")
                ax.set_xlabel("Tiempo")
                ax.set_ylabel("Valores Predichos")
                ax.set_ylim(y_min, y_max)
                ax.legend()
                plt.grid()
                st.pyplot(fig)
            else:
                st.error(f"No se encontró la columna '{prediction_col}' en las predicciones o no hay predicciones disponibles.")

            # Graficar valores reales
            resampled_data_sorted = resampled_data.sort_values(by=timestamp_column)
            context_length_plot = 128
            if len(resampled_data_sorted) > context_length_plot:
                real_data = resampled_data_sorted.iloc[-context_length_plot:]
            else:
                real_data = resampled_data_sorted

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(
                real_data[timestamp_column],
                real_data[target_column],
                label="Valor Real",
                linestyle="-",
                color="blue",
                linewidth=1,
            )
            ax.set_title("Valores Reales (Últimos 128 Registros)")
            ax.set_xlabel("Tiempo")
            ax.set_ylabel("Valores")
            ax.set_ylim(y_min, y_max)
            ax.legend()
            plt.grid()
            st.pyplot(fig)

            # Líneas fijas
            fixed_lines = [
                {"value": 230, "color": "orange", "label": "Límite 230"},
                {"value": 239, "color": "red", "label": "Límite 239"},
            ]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"###### Valores Reales (Últimos {context_length_plot} Registros) {selected_equipment}")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.plot(
                    real_data[timestamp_column],
                    real_data[target_column],
                    label="Valor Real",
                    linestyle="-",
                    color="blue",
                    linewidth=1,
                )
                for line in fixed_lines:
                    ax1.axhline(y=line["value"], color=line["color"], linestyle="--", linewidth=1, label=line["label"])
                ax1.set_title("Valores Reales", fontsize=12)
                ax1.set_xlabel("Tiempo", fontsize=10)
                ax1.set_ylabel("Valores", fontsize=10)
                ax1.set_ylim(y_min, y_max)
                ax1.legend(fontsize=8)
                plt.grid()
                st.pyplot(fig1)

            with col2:
                st.markdown(f"###### Predicciones (Horizonte 48 minutos) {selected_equipment}")
                if prediction_col in df_real_pred.columns and not df_pred.empty:
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.plot(
                        df_real_pred[timestamp_column],
                        df_real_pred[prediction_col],
                        label="Predicción",
                        linestyle="--",
                        color="green",
                    )
                    for line in fixed_lines:
                        ax2.axhline(y=line["value"], color=line["color"], linestyle="--", linewidth=1, label=line["label"])
                    ax2.set_title("Predicciones Generadas", fontsize=12)
                    ax2.set_xlabel("Tiempo", fontsize=10)
                    ax2.set_ylabel("Valores Predichos", fontsize=10)
                    ax2.set_ylim(y_min, y_max)
                    ax2.legend(fontsize=8)
                    plt.grid()
                    st.pyplot(fig2)
                else:
                    st.error(f"No se encontró la columna '{prediction_col}' en las predicciones o no hay predicciones disponibles.")

            # ====================================================================
            # Comparación de Predicciones Pasadas vs Valores Reales (Backtesting)
            # ====================================================================

            st.write("### Comparación de Predicciones Pasadas vs Valores Reales")

            # Realizar backtesting
            backtest_df = perform_backtesting(
                data=resampled_data,
                pipeline=pipeline,
                context_length=context_length,
                freq=freq,
                timestamp_column=timestamp_column,
                target_column=target_column,
                prediction_length=96,
                step=96
            )

            if not backtest_df.empty:
                st.write("### Tabla de Comparación de Predicciones vs Valores Reales")
                st.dataframe(backtest_df)

                # Calcular métricas de error
                mae = mean_absolute_error(backtest_df['Actual'], backtest_df['Predicted'])
                mse = mean_squared_error(backtest_df['Actual'], backtest_df['Predicted'])
                st.write(f"**MAE:** {mae:.2f}")
                st.write(f"**MSE:** {mse:.2f}")

                # Guardar en session_state
                st.session_state["real_vs_predicted"] = backtest_df
            else:
                st.info("No se generaron resultados de backtesting. Asegúrate de que hay suficientes datos.")
                st.stop()

        except Exception as e:
            st.error(f"Error durante la predicción: {e}")
else:
    st.error("No se pueden realizar predicciones porque el modelo o los escaladores no están cargados correctamente.")

# ====================================================================
# Mostrar Tabla de Error en el Tiempo (si está en session_state)
# ====================================================================

if "real_vs_predicted" in st.session_state and not st.session_state["real_vs_predicted"].empty:
    st.write("### Gráfico del Error en el Tiempo")
    real_vs_predicted = st.session_state["real_vs_predicted"].copy()
    real_vs_predicted["Error"] = real_vs_predicted["Actual"] - real_vs_predicted["Predicted"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(real_vs_predicted["Timestamp"], real_vs_predicted["Error"], color="orange", label="Error")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("Error entre Valores Reales y Predichos")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Error (°F)")
    ax.legend()
    plt.grid()
    st.pyplot(fig)

    # Opcional: Mostrar la tabla de errores
    st.write("### Tabla de Errores")
    st.dataframe(real_vs_predicted[['Timestamp', 'Error']])
